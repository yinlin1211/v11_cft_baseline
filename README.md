# CFHTransformer Baseline

当前仓库整理的是 `run/20260422_201016_COnP` 这次实验对应的代码、结果和独立后处理阈值搜索流程。

## 我们当前的结果

当前推荐汇报的是 MIR-ST500 测试集上的最终结果。

| Dataset | onset | frame | offset | COn | COnP | COnPOff |
|---------|------:|------:|-------:|----:|-----:|--------:|
| MIR-ST500 test set | 0.45 | 0.50 | 0.10 | 0.803172 | 0.776425 | 0.592620 |

这就是当前仓库里最完整、最推荐对外使用的一组结果。阈值搜索、训练过程和实验记录放在下面展开说明。

## 实验流程记录

### 1. 独立 val40 阈值搜索

第一阶段代码：

- `评估/search_threshold_v2.py`

运行命令：

```bash
python3 评估/search_threshold_v2.py \
  --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
  --output_dir run/20260422_201016_COnP/threshold_search_v2_epoch0128
```

第一阶段结果：

- `BEST_VAL_BY_COnP onset=0.45 frame=0.50`
- `val COn=0.817416`
- `val COnP=0.802107`
- `val COnPOff=0.529928`
- `test COn=0.804001`
- `test COnP=0.777067`
- `test COnPOff=0.482631`

第二阶段代码：

- `评估/search_offset_threshold_and_predict.py`

运行命令：

```bash
python3 评估/search_offset_threshold_and_predict.py \
  --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
  --onset_thresh 0.45 \
  --frame_thresh 0.50 \
  --output_dir run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp
```

第二阶段结果：

- `SELECTED_BY_VAL off=0.10`
- `test COn=0.803172`
- `test COnP=0.776425`
- `test COnPOff=0.592620`

### 2. 结果文件

- `run/20260422_201016_COnP/threshold_search_v2_epoch0128/val_threshold_search.tsv`
- `run/20260422_201016_COnP/threshold_search_v2_epoch0128/selected_thresholds.tsv`
- `run/20260422_201016_COnP/threshold_search_v2_epoch0128/test_with_selected_thresholds.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/val_offset_threshold_search.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/selected_offset_threshold.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/test_with_selected_offset_threshold.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/pred_test_offset_aware.json`

### 3. 训练内阈值与独立搜索的区别

训练脚本里保存过一组训练内阈值：

- `onset=0.50`
- `frame=0.40`

这组阈值来自训练过程中每 5 个 epoch 的 val 搜索，并记录在 checkpoint / monitor 里。

但独立评估脚本重新按固定口径搜索后，得到的是：

- `onset=0.45`
- `frame=0.50`

因此对外报告时，优先建议使用 `评估/` 目录下独立搜索得到的完整实验结果。

## 代码位置

- `train_conp_v6_0415.py`：训练脚本，包含训练内阈值搜索与 test monitor
- `predict_to_json.py`：baseline 推理
- `predict_to_json_offset.py`：offset-aware 推理
- `评估/search_threshold_v2.py`：独立 `val40` 搜 `onset/frame`
- `评估/search_offset_threshold_and_predict.py`：固定 `onset/frame` 后独立 `val40` 搜 `offset`
- `evaluate_github.py`：原论文兼容评测脚本

## 环境与数据

- Python 3.12
- PyTorch
- librosa
- mir_eval
- CQT 缓存：288-bin, hop=800, 50ms/帧
- 标注：`data/MIR-ST500_corrected.json`
- 划分：`splits_v11/` (`train=1-400`, `val=361-400`, `test=401-500`)

## 模型权重

仓库不上传 `.pt` 权重文件，只保留代码、结果、日志和阈值表。

## 评测脚本来源

`evaluate_github.py` 来源于 [york135/singing_transcription_ICASSP2021](https://github.com/york135/singing_transcription_ICASSP2021/tree/master/evaluate)。
