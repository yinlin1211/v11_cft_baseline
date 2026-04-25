import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from model import CFT_v6  # noqa: E402
from predict_to_json import frames_to_notes, predict_from_npy  # noqa: E402
from predict_to_json_offset import frames_to_notes_offset  # noqa: E402
from train_conp_v6_0415 import compute_note_f1_single  # noqa: E402


CONFIG_PATH = ROOT_DIR / "config.yaml"
GPU_IDS = [1, 2, 3]
CPU_WORKERS = max(1, os.cpu_count() or 1)

GLOBAL_PREDS = None
GLOBAL_CONFIG = None
GLOBAL_ONSET = None
GLOBAL_FRAME = None


def log(message, log_file=None):
    print(message, flush=True)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def notes_to_arrays(notes):
    if not notes:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    return (
        np.array([[float(n[0]), float(n[1])] for n in notes], dtype=float),
        np.array([float(n[2]) for n in notes], dtype=float),
    )


def build_thresholds(start, stop, step):
    return [round(float(x), 2) for x in np.arange(start, stop + step / 2.0, step)]


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(onset, frame, metrics):
    return {
        "onset_thresh": f"{onset:.2f}",
        "frame_thresh": f"{frame:.2f}",
        "COn_f1": f"{metrics['COn']:.9f}",
        "COnP_f1": f"{metrics['COnP']:.9f}",
        "COnPOff_f1": f"{metrics['COnPOff']:.9f}",
        "COn_plus_COnP": f"{metrics['COn'] + metrics['COnP']:.9f}",
        "sum_all": f"{metrics['COn'] + metrics['COnP'] + metrics['COnPOff']:.9f}",
    }


def metric_row_offset(offset, metrics, onset, frame):
    return {
        "onset_thresh": f"{onset:.2f}",
        "frame_thresh": f"{frame:.2f}",
        "offset_thresh": f"{offset:.2f}",
        "COn_f1": f"{metrics['COn']:.9f}",
        "COnP_f1": f"{metrics['COnP']:.9f}",
        "COnPOff_f1": f"{metrics['COnPOff']:.9f}",
        "COn_plus_COnP": f"{metrics['COn'] + metrics['COnP']:.9f}",
        "sum_all": f"{metrics['COn'] + metrics['COnP'] + metrics['COnPOff']:.9f}",
    }


def chunk_list(items, n):
    chunks = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        chunks[idx % n].append(item)
    return chunks


def infer_worker(gpu_id, song_ids, cache_dir_str, checkpoint_str, config_path_str):
    cache_dir = Path(cache_dir_str)
    with open(config_path_str) as f:
        config = yaml.safe_load(f)
    device = torch.device(f"cuda:{gpu_id}")
    model = CFT_v6(config).to(device)
    ckpt = torch.load(checkpoint_str, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    npy_dir = Path(config["data"]["cqt_cache_dir"])
    started = time.time()
    for idx, song_id in enumerate(song_ids):
        frame_prob, onset_prob, offset_prob = predict_from_npy(
            model, str(npy_dir / f"{song_id}.npy"), config, device
        )
        np.savez_compressed(
            cache_dir / f"{song_id}.npz",
            frame_prob=frame_prob,
            onset_prob=onset_prob,
            offset_prob=offset_prob,
        )
        print(f"[GPU{gpu_id}] infer [{idx + 1:3d}/{len(song_ids)}] song {song_id}", flush=True)
    print(f"[GPU{gpu_id}] infer done in {time.time() - started:.1f}s", flush=True)


def infer_parallel(song_ids, cache_dir, checkpoint_path, gpu_ids, log_file):
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_list(song_ids, len(gpu_ids))
    log(f"Start GPU inference cache -> {cache_dir}", log_file)
    log(f"Use GPUs: {gpu_ids}", log_file)
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=infer_worker,
            args=(gpu_id, chunk, str(cache_dir), str(checkpoint_path), str(CONFIG_PATH)),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"inference worker failed with exit code {p.exitcode}")
    log(f"Finish GPU inference cache -> {cache_dir}", log_file)


def load_cached_predictions(cache_dir, gt_annotations):
    preds = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        song_id = npz_path.stem
        data = np.load(npz_path)
        raw = gt_annotations[song_id]
        notes = [
            [float(n[0]), float(n[1]), float(n[2])]
            for n in raw
            if float(n[1]) - float(n[0]) > 0
        ]
        ref_intervals, ref_pitches = notes_to_arrays(notes)
        preds.append(
            (
                song_id,
                data["frame_prob"],
                data["onset_prob"],
                data["offset_prob"],
                ref_intervals,
                ref_pitches,
            )
        )
    return preds


def pool_init_stage1(preds, config, fixed_frame):
    global GLOBAL_PREDS, GLOBAL_CONFIG, GLOBAL_FRAME
    GLOBAL_PREDS = preds
    GLOBAL_CONFIG = config
    GLOBAL_FRAME = fixed_frame


def score_stage1_onset(onset_thresh):
    hop_length = GLOBAL_CONFIG["audio"]["hop_length"]
    sample_rate = GLOBAL_CONFIG["data"]["sample_rate"]
    con_scores, conp_scores, conpoff_scores = [], [], []
    for _, frame_prob, onset_prob, _, ref_intervals, ref_pitches in GLOBAL_PREDS:
        notes = frames_to_notes(
            frame_prob,
            onset_prob,
            hop_length,
            sample_rate,
            onset_thresh=onset_thresh,
            frame_thresh=GLOBAL_FRAME,
        )
        pred_intervals, pred_pitches = notes_to_arrays(notes)
        con, conp, conpoff = compute_note_f1_single(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches
        )
        if conp is not None:
            con_scores.append(con)
            conp_scores.append(conp)
            conpoff_scores.append(conpoff)
    return {
        "onset": onset_thresh,
        "frame": GLOBAL_FRAME,
        "COn": float(np.mean(con_scores)) if con_scores else 0.0,
        "COnP": float(np.mean(conp_scores)) if conp_scores else 0.0,
        "COnPOff": float(np.mean(conpoff_scores)) if conpoff_scores else 0.0,
    }


def pool_init_stage1_frame(preds, config, fixed_onset):
    global GLOBAL_PREDS, GLOBAL_CONFIG, GLOBAL_ONSET
    GLOBAL_PREDS = preds
    GLOBAL_CONFIG = config
    GLOBAL_ONSET = fixed_onset


def score_stage1_frame(frame_thresh):
    hop_length = GLOBAL_CONFIG["audio"]["hop_length"]
    sample_rate = GLOBAL_CONFIG["data"]["sample_rate"]
    con_scores, conp_scores, conpoff_scores = [], [], []
    for _, frame_prob, onset_prob, _, ref_intervals, ref_pitches in GLOBAL_PREDS:
        notes = frames_to_notes(
            frame_prob,
            onset_prob,
            hop_length,
            sample_rate,
            onset_thresh=GLOBAL_ONSET,
            frame_thresh=frame_thresh,
        )
        pred_intervals, pred_pitches = notes_to_arrays(notes)
        con, conp, conpoff = compute_note_f1_single(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches
        )
        if conp is not None:
            con_scores.append(con)
            conp_scores.append(conp)
            conpoff_scores.append(conpoff)
    return {
        "onset": GLOBAL_ONSET,
        "frame": frame_thresh,
        "COn": float(np.mean(con_scores)) if con_scores else 0.0,
        "COnP": float(np.mean(conp_scores)) if conp_scores else 0.0,
        "COnPOff": float(np.mean(conpoff_scores)) if conpoff_scores else 0.0,
    }


def pool_init_stage2(preds, config, onset_thresh, frame_thresh):
    global GLOBAL_PREDS, GLOBAL_CONFIG, GLOBAL_ONSET, GLOBAL_FRAME
    GLOBAL_PREDS = preds
    GLOBAL_CONFIG = config
    GLOBAL_ONSET = onset_thresh
    GLOBAL_FRAME = frame_thresh


def score_stage2_offset(offset_thresh):
    hop_length = GLOBAL_CONFIG["audio"]["hop_length"]
    sample_rate = GLOBAL_CONFIG["data"]["sample_rate"]
    con_scores, conp_scores, conpoff_scores = [], [], []
    for _, frame_prob, onset_prob, offset_prob, ref_intervals, ref_pitches in GLOBAL_PREDS:
        notes = frames_to_notes_offset(
            frame_prob,
            onset_prob,
            offset_prob,
            hop_length,
            sample_rate,
            onset_thresh=GLOBAL_ONSET,
            frame_thresh=GLOBAL_FRAME,
            offset_thresh=offset_thresh,
        )
        pred_intervals, pred_pitches = notes_to_arrays(notes)
        con, conp, conpoff = compute_note_f1_single(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches
        )
        if conp is not None:
            con_scores.append(con)
            conp_scores.append(conp)
            conpoff_scores.append(conpoff)
    return {
        "offset": offset_thresh,
        "COn": float(np.mean(con_scores)) if con_scores else 0.0,
        "COnP": float(np.mean(conp_scores)) if conp_scores else 0.0,
        "COnPOff": float(np.mean(conpoff_scores)) if conpoff_scores else 0.0,
    }


def parallel_search(pool_init, fn, items, workers, stage_name, best_key_fn, log_file):
    total = len(items)
    results = []
    started = time.time()
    log(f"Start {stage_name}: {total} combinations, workers={workers}", log_file)
    with mp.Pool(processes=min(workers, total), initializer=pool_init[0], initargs=pool_init[1]) as pool:
        for idx, result in enumerate(pool.imap_unordered(fn, items, chunksize=2), start=1):
            results.append(result)
            best_so_far = max(results, key=best_key_fn)
            log(
                f"[{stage_name}] {idx}/{total} done | current best = {best_so_far} | elapsed={time.time() - started:.1f}s",
                log_file,
            )
    return results


def score_test(preds, onset_thresh, frame_thresh, offset_thresh, config):
    hop_length = config["audio"]["hop_length"]
    sample_rate = config["data"]["sample_rate"]
    con_scores, conp_scores, conpoff_scores = [], [], []
    pred_json = {}
    for song_id, frame_prob, onset_prob, offset_prob, ref_intervals, ref_pitches in preds:
        notes = frames_to_notes_offset(
            frame_prob,
            onset_prob,
            offset_prob,
            hop_length,
            sample_rate,
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
            offset_thresh=offset_thresh,
        )
        pred_json[song_id] = notes
        pred_intervals, pred_pitches = notes_to_arrays(notes)
        con, conp, conpoff = compute_note_f1_single(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches
        )
        if conp is not None:
            con_scores.append(con)
            conp_scores.append(conp)
            conpoff_scores.append(conpoff)
    return {
        "COn": float(np.mean(con_scores)) if con_scores else 0.0,
        "COnP": float(np.mean(conp_scores)) if conp_scores else 0.0,
        "COnPOff": float(np.mean(conpoff_scores)) if conpoff_scores else 0.0,
        "pred_json": pred_json,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--gpus", default="1,2,3")
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--fixed-frame", type=float, required=True, help="frame threshold used in first onset sweep")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    workers = max(1, args.cpu_workers)
    fixed_frame = args.fixed_frame

    epoch_tag = checkpoint_path.stem.replace("best_model_epoch", "epoch").split("_COnP")[0]
    frame_tag = str(fixed_frame).replace(".", "")
    base_out = THIS_DIR / f"fast_cpu_priority_{epoch_tag}_ff{frame_tag}"
    val_cache_dir = base_out / "cache_val"
    test_cache_dir = base_out / "cache_test"
    stage1_out = base_out / f"fast_threshold_search_{epoch_tag}"
    stage2_out = base_out / f"fast_offset_search_{epoch_tag}"
    log_path = base_out / "run.log"

    if base_out.exists():
        shutil.rmtree(base_out)
    stage1_out.mkdir(parents=True, exist_ok=True)
    stage2_out.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_file:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        with open(config["data"]["label_path"]) as f:
            gt_annotations = json.load(f)
        splits_dir = Path(config["data"]["splits_dir"])
        with open(splits_dir / "val.txt") as f:
            val_song_ids = [line.strip() for line in f if line.strip()]
        with open(splits_dir / "test.txt") as f:
            test_song_ids = [line.strip() for line in f if line.strip()]

        log(f"checkpoint: {checkpoint_path}", log_file)
        log("fast flow: fixed frame -> onset search -> frame search -> offset search -> test", log_file)
        log(f"gpu ids for cache: {gpu_ids}", log_file)
        log(f"cpu workers for threshold search: {workers}", log_file)
        log(f"fixed frame for onset sweep: {fixed_frame:.2f}", log_file)

        infer_parallel(val_song_ids, val_cache_dir, checkpoint_path, gpu_ids, log_file)
        val_preds = load_cached_predictions(val_cache_dir, gt_annotations)

        onset_candidates = build_thresholds(0.05, 1.00, 0.05)
        onset_rows = parallel_search(
            (pool_init_stage1, (val_preds, config, fixed_frame)),
            score_stage1_onset,
            onset_candidates,
            workers,
            "stage1_onset",
            lambda r: (r["COnP"], r["COn"], r["COnPOff"]),
            log_file,
        )
        best_onset = max(onset_rows, key=lambda r: (r["COnP"], r["COn"], r["COnPOff"]))
        onset_thresh = best_onset["onset"]
        log(
            f"STAGE1_ONSET_SELECTED onset={onset_thresh:.2f} fixed_frame={fixed_frame:.2f} "
            f"val_COn={best_onset['COn']:.6f} val_COnP={best_onset['COnP']:.6f} val_COnPOff={best_onset['COnPOff']:.6f}",
            log_file,
        )

        frame_candidates = build_thresholds(0.05, 1.00, 0.05)
        frame_rows = parallel_search(
            (pool_init_stage1_frame, (val_preds, config, onset_thresh)),
            score_stage1_frame,
            frame_candidates,
            workers,
            "stage1_frame",
            lambda r: (r["COnP"], r["COn"], r["COnPOff"]),
            log_file,
        )
        best_frame = max(frame_rows, key=lambda r: (r["COnP"], r["COn"], r["COnPOff"]))
        frame_thresh = best_frame["frame"]
        log(
            f"STAGE1_FRAME_SELECTED onset={onset_thresh:.2f} frame={frame_thresh:.2f} "
            f"val_COn={best_frame['COn']:.6f} val_COnP={best_frame['COnP']:.6f} val_COnPOff={best_frame['COnPOff']:.6f}",
            log_file,
        )

        fields = [
            "onset_thresh",
            "frame_thresh",
            "COn_f1",
            "COnP_f1",
            "COnPOff_f1",
            "COn_plus_COnP",
            "sum_all",
        ]
        onset_stage_rows = [metric_row(r["onset"], fixed_frame, r) for r in onset_rows]
        write_rows(stage1_out / "onset_search.tsv", fields, onset_stage_rows)
        frame_stage_rows = [metric_row(onset_thresh, r["frame"], r) for r in frame_rows]
        write_rows(stage1_out / "frame_search.tsv", fields, frame_stage_rows)
        selected_rows = [
            {"criterion": "best_COnP_after_fast_search", **metric_row(onset_thresh, frame_thresh, best_frame)}
        ]
        write_rows(stage1_out / "selected_thresholds.tsv", ["criterion"] + fields, selected_rows)

        offset_candidates = build_thresholds(0.05, 1.00, 0.05)
        offset_rows = parallel_search(
            (pool_init_stage2, (val_preds, config, onset_thresh, frame_thresh)),
            score_stage2_offset,
            offset_candidates,
            workers,
            "stage2_offset",
            lambda r: (r["COnPOff"], r["COnP"], r["COn"]),
            log_file,
        )
        best_offset = max(offset_rows, key=lambda r: (r["COnPOff"], r["COnP"], r["COn"]))
        offset_thresh = best_offset["offset"]
        log(
            f"STAGE2_SELECTED onset={onset_thresh:.2f} frame={frame_thresh:.2f} offset={offset_thresh:.2f} "
            f"val_COn={best_offset['COn']:.6f} val_COnP={best_offset['COnP']:.6f} val_COnPOff={best_offset['COnPOff']:.6f}",
            log_file,
        )

        fields_offset = [
            "onset_thresh",
            "frame_thresh",
            "offset_thresh",
            "COn_f1",
            "COnP_f1",
            "COnPOff_f1",
            "COn_plus_COnP",
            "sum_all",
        ]
        offset_stage_rows = [metric_row_offset(r["offset"], r, onset_thresh, frame_thresh) for r in offset_rows]
        write_rows(stage2_out / "val_offset_threshold_search.tsv", fields_offset, offset_stage_rows)
        selected_offset_row = {
            "criterion": "best_val_COnPOff_after_fast_search",
            **metric_row_offset(offset_thresh, best_offset, onset_thresh, frame_thresh),
        }
        write_rows(stage2_out / "selected_offset_threshold.tsv", ["criterion"] + fields_offset, [selected_offset_row])

        infer_parallel(test_song_ids, test_cache_dir, checkpoint_path, gpu_ids, log_file)
        test_preds = load_cached_predictions(test_cache_dir, gt_annotations)
        test_metrics = score_test(test_preds, onset_thresh, frame_thresh, offset_thresh, config)
        test_row = {"criterion": "test_with_val_selected_offset"}
        test_row.update(metric_row_offset(offset_thresh, test_metrics, onset_thresh, frame_thresh))
        write_rows(stage2_out / "test_with_selected_offset_threshold.tsv", ["criterion"] + fields_offset, [test_row])
        with (stage2_out / "pred_test_offset_aware.json").open("w") as f:
            json.dump(test_metrics["pred_json"], f, indent=2, ensure_ascii=False)

        log(
            f"FINAL_TEST onset={onset_thresh:.2f} frame={frame_thresh:.2f} offset={offset_thresh:.2f} "
            f"COn={test_metrics['COn']:.6f} COnP={test_metrics['COnP']:.6f} "
            f"COnPOff={test_metrics['COnPOff']:.6f}",
            log_file,
        )
        log(f"outputs: {base_out}", log_file)


if __name__ == "__main__":
    main()
