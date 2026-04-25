from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt


ROOT = Path("/mnt/ssd/lian/给claudecode/v10_baseline_conpoff/用其他的轮计算")
LOG_PATH = ROOT / "batch_fast_80_500.log"
CSV_PATH = ROOT / "batch_fast_80_500_summary.csv"
PNG_PATH = ROOT / "batch_fast_80_500_metrics.png"


def parse_results(log_path: Path):
    lines = log_path.read_text(errors="ignore").splitlines()
    start_pat = re.compile(r"^\[(\d+)/(\d+)\] START (best_model_epoch(\d+)_COnP[\d.]+\.pt)")
    final_pat = re.compile(
        r"FINAL_TEST onset=([\d.]+) frame=([\d.]+) offset=([\d.]+) "
        r"COn=([\d.]+) COnP=([\d.]+) COnPOff=([\d.]+)"
    )

    current = None
    rows = []
    for line in lines:
        m = start_pat.search(line)
        if m:
            current = {
                "checkpoint": m.group(3),
                "epoch": int(m.group(4)),
            }
            continue
        m = final_pat.search(line)
        if m and current:
            rows.append(
                {
                    "epoch": current["epoch"],
                    "checkpoint": current["checkpoint"],
                    "onset": float(m.group(1)),
                    "frame": float(m.group(2)),
                    "offset": float(m.group(3)),
                    "COn": float(m.group(4)),
                    "COnP": float(m.group(5)),
                    "COnPOff": float(m.group(6)),
                }
            )
    rows.sort(key=lambda x: x["epoch"])
    return rows


def save_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "checkpoint", "onset", "frame", "offset", "COn", "COnP", "COnPOff"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path: Path):
    epochs = [r["epoch"] for r in rows]
    con = [r["COn"] for r in rows]
    conp = [r["COnP"] for r in rows]
    conpoff = [r["COnPOff"] for r in rows]

    best_con_idx = max(range(len(rows)), key=lambda i: con[i])
    best_conp_idx = max(range(len(rows)), key=lambda i: conp[i])
    best_conpoff_idx = max(range(len(rows)), key=lambda i: conpoff[i])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
    )

    ax1.plot(epochs, con, marker="o", linewidth=2.4, markersize=5, label="COn")
    ax1.plot(epochs, conp, marker="s", linewidth=2.4, markersize=5, label="COnP")
    ax1.scatter([epochs[best_con_idx]], [con[best_con_idx]], s=80, zorder=5)
    ax1.scatter([epochs[best_conp_idx]], [conp[best_conp_idx]], s=80, zorder=5)
    ax1.annotate(f"best COn @{epochs[best_con_idx]}", (epochs[best_con_idx], con[best_con_idx]),
                 textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax1.annotate(f"best COnP @{epochs[best_conp_idx]}", (epochs[best_conp_idx], conp[best_conp_idx]),
                 textcoords="offset points", xytext=(6, -14), fontsize=9)
    ax1.set_ylabel("F1")
    ax1.set_title("Batch Fast Search Results (Epoch 80-500): COn and COnP")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(epochs, conpoff, color="tab:green", marker="^", linewidth=2.4, markersize=5, label="COnPOff")
    ax2.scatter([epochs[best_conpoff_idx]], [conpoff[best_conpoff_idx]], s=90, zorder=5, color="tab:red")
    ax2.annotate(f"best COnPOff @{epochs[best_conpoff_idx]}", (epochs[best_conpoff_idx], conpoff[best_conpoff_idx]),
                 textcoords="offset points", xytext=(6, 8), fontsize=9)
    ymin = min(conpoff) - 0.01
    ymax = max(conpoff) + 0.01
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel("Checkpoint Epoch")
    ax2.set_ylabel("COnPOff")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    rows = parse_results(LOG_PATH)
    if not rows:
        raise RuntimeError(f"No results parsed from {LOG_PATH}")
    save_csv(rows, CSV_PATH)
    plot(rows, PNG_PATH)
    print(f"saved csv: {CSV_PATH}")
    print(f"saved plot: {PNG_PATH}")


if __name__ == "__main__":
    main()
