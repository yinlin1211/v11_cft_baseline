# CFHTransformer Baseline

本仓库当前以 `run/20260422_201016_COnP` 这次训练及其后续推理实验为准。这里把代码入口、阈值来源、关键结果和可复现命令整理到一处，避免再混用旧版 v7 说明。

## 当前 baseline 口径

- 模型 GAP：当前代码与现有 checkpoint 固定使用 `S.mean(dim=-1)`；如果要尝试 `sum + LayerNorm`，需要改模型并重新训练。
- 数据划分：`train=1-400`，`val=361-400`，`test=401-500`。因此 val 与 train 有 40 首重叠，val 只能作为训练内监控。
- 当前训练记录对应的最佳权重来自 `run/20260422_201016_COnP/checkpoints/`，但模型权重文件未上传到本仓库。
- `epoch0128` 是后续 offset-aware 推理时反复使用的一版 checkpoint；本仓库保留了对应结果和命令说明，但不附带 `.pt` 文件。
- `test_monitor.txt` 使用训练脚本的全曲 chunk 评估；`predict_to_json.py` / `predict_to_json_offset.py` 使用 50% overlap 推理，因此两者 test 数值不完全等价。

## 代码入口

- `model.py`：模型与 loss 定义。
- `dataset.py`：数据集读取与标签构造。
- `train_conp_v6_0415.py`：训练主脚本，内置 val 阈值搜索与 test monitor。
- `predict_to_json.py`：baseline 推理脚本，输出与原评测脚本兼容的 JSON。
- `predict_to_json_offset.py`：offset-aware 后处理推理脚本，使用 onset/frame/offset 三个分支解码。
- `evaluate_github.py`：原论文兼容评测脚本。
- `config.yaml`：当前 baseline 配置。

## 上传范围

- 已上传：代码、配置、划分文件、评测脚本、结果 JSON、训练监控表、阈值搜索表、训练 stdout 日志。
- 未上传：模型权重 `.pt` 文件。

## 阈值是怎么来的

### 1. `onset=0.50`, `frame=0.40`

这两个阈值来自训练脚本里的 val 阈值搜索：

- 搜索代码：`train_conp_v6_0415.py` 中的 `find_best_threshold()`
- 搜索网格：`onset_thresholds x frame_thresholds x offset_thresholds`
- 触发方式：训练时每 5 个 epoch 在 val 上搜索一次 best threshold

训练命令：

```bash
cd /mnt/ssd/lian/给claudecode/v10_baseline_conpoff
CUDA_VISIBLE_DEVICES=1 python3 train_conp_v6_0415.py --config config.yaml
```

已能直接确认的结果：

- `run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt`
  - `epoch=128`
  - `COn_f1=0.812787`
  - `COnP_f1=0.795783`
  - `COnPOff_f1=0.486836`
  - `best_onset_thresh=0.50`
  - `best_frame_thresh=0.40`
- `run/20260422_201016_COnP/test_monitor.txt` 中 epoch 128 的 full-test monitor：
  - `COn_f1=0.799798`
  - `COnP_f1=0.770271`
  - `COnPOff_f1=0.440881`
  - `onset_thresh=0.50`
  - `frame_thresh=0.40`

### 2. `offset=0.20`

当前仓库里没有留下单独的 val offset 搜索表，但保留了 offset-aware 推理结果文件。现有证据表明，后处理阶段是固定 `epoch0128 + onset=0.50 + frame=0.40`，再手动试不同 `offset_thresh` 得到的。

使用的推理脚本：

```bash
cd /mnt/ssd/lian/给claudecode/v10_baseline_conpoff
python3 predict_to_json_offset.py \
  --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
  --split test \
  --onset_thresh 0.50 \
  --frame_thresh 0.40 \
  --offset_thresh 0.20 \
  --output pred_test_epoch0128_offset_off020.json

python3 evaluate_github.py \
  data/MIR-ST500_corrected.json \
  pred_test_epoch0128_offset_off020.json \
  0.05
```

对应结果：

- `COn = 0.801662`
- `COnP = 0.774792`
- `COnPOff = 0.566972`

同一脚本下，`offset=0.30` 的对照命令：

```bash
python3 predict_to_json_offset.py \
  --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
  --split test \
  --onset_thresh 0.50 \
  --frame_thresh 0.40 \
  --offset_thresh 0.30 \
  --output pred_test_epoch0128_offset_off030.json

python3 evaluate_github.py \
  data/MIR-ST500_corrected.json \
  pred_test_epoch0128_offset_off030.json \
  0.05
```

对应结果：

- `COn = 0.801525`
- `COnP = 0.774814`
- `COnPOff = 0.586965`

## 现有预测文件复评

统一评测命令：

```bash
python3 evaluate_github.py data/MIR-ST500_corrected.json <pred.json> 0.05
```

| 预测文件 | 说明 | COn | COnP | COnPOff | pred notes |
|----------|------|----:|-----:|--------:|-----------:|
| `pred_test_v7.json` | 旧版 baseline 结果 | 0.799389 | 0.772119 | 0.425381 | 31203 |
| `pred_test_v10_best.json` | 当前 repo 下 `best_model.pt` 的 baseline 推理 | 0.788769 | 0.762855 | 0.457117 | 30589 |
| `run/20260422_201016_COnP/pred_test_best_model_epoch0316_best_grid.json` | 在 test 上做阈值网格后的结果 | 0.797590 | 0.771582 | 0.469622 | 30121 |
| `pred_test_epoch0128_offset_off020.json` | `epoch0128 + on=0.50 + fr=0.40 + off=0.20` | 0.801662 | 0.774792 | 0.566972 | 30683 |
| `pred_test_epoch0128_offset_off030.json` | `epoch0128 + on=0.50 + fr=0.40 + off=0.30` | 0.801525 | 0.774814 | 0.586965 | 30712 |

说明：

- `epoch0316_best_grid` 是在 test 上做阈值网格后得到的结果，适合分析上限，不应作为无偏选模指标。
- `pred_test_epoch0128_offset_off020.json` 是仓库里现成可复现、并且与当时记录 `COn≈80.16 / COnP≈77.47 / COnPOff≈0.5669` 对上的那一份。
- `pred_test_epoch0128_offset_off030.json` 的 `COnPOff` 更高，但它不是当前仓库里留有明确 val 选优记录的一份。

## 当前推荐结果

如果按“当前仓库里已有代码和文件，能明确复现并解释来源”的标准，推荐保留下面这组说明：

- baseline 两阈值来自训练阶段 val 搜索：`onset=0.50`, `frame=0.40`
- 后处理 offset-aware 推理基于 `epoch0128` checkpoint
- 当前已验证的 offset-aware 结果：
  - `off=0.20`: `COn=0.801662`, `COnP=0.774792`, `COnPOff=0.566972`
  - `off=0.30`: `COn=0.801525`, `COnP=0.774814`, `COnPOff=0.586965`

## 环境与数据依赖

- Python 3.12, PyTorch, librosa, mir_eval
- CQT 预计算缓存：288-bin, hop=800, 50ms/帧
- 标注文件：`data/MIR-ST500_corrected.json`
- 数据集划分：`splits_v11/` (`train=1-400`, `val=361-400`, `test=401-500`)

## 训练产物

训练产物保存在 `run/<timestamp>_COnP/` 下。当前主要目录：

- `run/20260422_201016_COnP/test_monitor.txt`：full test 监控记录
- `run/20260422_201016_COnP/threshold_search_test_best_model_epoch0316.tsv`：一份 test 阈值网格结果表
- `run/20260422_201016_COnP/logs/train_stdout.log`：训练日志

说明：`checkpoints/*.pt` 未上传。

## 评测脚本来源

`evaluate_github.py` 来源于 [york135/singing_transcription_ICASSP2021](https://github.com/york135/singing_transcription_ICASSP2021/tree/master/evaluate)。
