# PAOT2 CT ICL 最小训练

这条流水线只用于验证 CCTI/BCI channel selection 想法，不代表最终临床训练协议。
它将每个 3D CT 和任务二值 mask 整体缩放为固定立方体，避免在验证时用真值位置裁剪。

## 标签定义

完整机器可读规则见 `configs/paot2_task_map.json`。

| 路径任务 | 器官 | 二值肿瘤标签 |
|---|---|---|
| LiTS、MSD Task03 | liver | 2 |
| KiTS | kidney | 2 |
| MSD Task06 | lung | 1 |
| MSD Task07、PanTS | pancreas | 2 |
| MSD Task08 | liver | 2 |
| MSD Task10 | colon | 1 |
| AbdomenCT-1K `03/05/07` | liver/kidney/pancreas | 5/6/7 |
| MSWAL `03/05/07` | liver/kidney/pancreas | 3/4/5 |

带有多个前缀的 AbdomenCT-1K 或 MSWAL 病例会被展开成多个任务 episode。每个
episode 只生成一个器官肿瘤二值 mask。context 只从 train 中选择，且必须与 target
的 `target_region` 相同、患者不同、肿瘤非空。

## 1. 生成并检查 manifest

在服务器工程目录执行：

```bash
cd /path/to/Medverse
python scripts/prepare_paot2_manifest.py \
  --train-pairs /path/to/PAOT2_train.txt \
  --val-pairs /path/to/PAOT2_val.txt \
  --data-root /public_bme2/bme-dgshen/RunqiMeng/project \
  --inspect-labels \
  --output /private/workdir/paot2_icl.jsonl

python scripts/validate_pan_cancer_manifest.py /private/workdir/paot2_icl.jsonl
```

`--inspect-labels` 会确认路径存在、读取 NIfTI 缩放系数、统计每个任务的肿瘤体素，
并默认删除该任务肿瘤为空的 episode。生成的 manifest 含绝对服务器路径，不应提交
GitHub。

## 2. 一步训练 smoke test

先用小模型和 64³ 输入验证加载、前向、反向及验证：

```bash
python scripts/train_pan_cancer_icl.py \
  --manifest /private/workdir/paot2_icl.jsonl \
  --output-dir /private/workdir/runs/smoke \
  --image-size 64 \
  --channels 8,16,32,64,128 \
  --num-context 1 \
  --batch-size 1 \
  --workers 2 \
  --epochs 1 \
  --max-train-steps 1 \
  --max-val-steps 1
```

成功标准是产生一条 train/val 指标记录以及 `last.pt`、`best.pt`。

## 3. MVP 训练与测试

```bash
python scripts/train_pan_cancer_icl.py \
  --manifest /private/workdir/paot2_icl.jsonl \
  --output-dir /private/workdir/runs/learned_seed17 \
  --image-size 128 \
  --num-context 2 \
  --batch-size 1 \
  --workers 4 \
  --epochs 20 \
  --ccti-mode learned \
  --ccti-ratio 0.25 \
  --seed 17

python scripts/train_pan_cancer_icl.py \
  --manifest /private/workdir/paot2_icl.jsonl \
  --checkpoint /private/workdir/runs/learned_seed17/best.pt \
  --eval-only
```

脚本输出总体 macro Dice 和各器官肿瘤 Dice。正式比较时仅改变 `--ccti-mode`：
`none`、`all`、`random`、`learned`，其余参数、manifest 和随机种子保持一致。
