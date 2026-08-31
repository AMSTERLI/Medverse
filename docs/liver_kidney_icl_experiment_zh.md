# 肝/肾肿瘤 No-ICL 与 ICL+BAM 实验

## 冻结的研究设计

只使用 LiTS、KiTS23、MSWAL，任务只有 `liver_tumor` 和
`kidney_tumor`。标签映射固定为 LiTS 2、KiTS23 2、MSWAL 3/4；KiTS23
囊肿 3 和 MSWAL 其他病变不并入肿瘤。划分种子为 `20260831`，按
`source_dataset + patient_id` 分组并按数据源/任务分层，比例 70/10/20。

TotalSegmentator 2.18.0 是冻结的离线器官定位器，只推理 `liver`、
`kidney_left`、`kidney_right`。肝使用肝 bbox，肾使用左右肾并集 bbox，
各方向物理外扩 30 mm。裁剪阶段不重采样、不读肿瘤位置；保存 RAS
方向、原 affine、bbox 和逆变换。nnU-Net v2 fingerprint/planner 决定共享
spacing；该 spacing 快照再供 Medverse 使用。

主比较预先固定为：

- GPU 0：共享二值 nnU-Net v2 `3d_fullres`，输入 CT + 器官 mask，无 ICL。
- GPU 1：PanCancerMedverse，输入同样两通道，K=3 固定同任务训练集
  context，启用原始 BAM，`use_ccti=false`。

第一轮不做 5-fold。任何时刻最多两个训练进程，不使用两卡 DDP；只有
两个 2–4 病例过拟合门禁均通过后，正式双模型启动脚本才允许运行。

## 环境

推荐新建独立 Python 3.10 环境，不修改旧项目环境。先安装与集群 CUDA
匹配的 PyTorch，再安装：

```bash
python -m pip install -r requirements-liver-kidney.txt
python -m pip check
```

CentOS 7 若最新版 `connected-components-3d`/`blosc2` 尝试源码编译，使用
本仓库固定的 manylinux2014 兼容版本。验证命令：

```bash
python -c "import torch, nnunetv2, totalsegmentator; print(torch.__version__, torch.cuda.is_available())"
nnUNetv2_plan_and_preprocess -h
TotalSegmentator --help
pytest -q tests/test_liver_kidney_pipeline.py tests/test_context_and_evaluation.py
```

## 阶段 1：标签审计、固定划分与器官定位

所有路径通过环境变量或参数传入，以下仅为相对路径示例：

```bash
python scripts/prepare_main_experiment_manifest.py \
  --input work/paot2_icl_inspected.jsonl \
  --output work/data/derived_liver_kidney_v1/manifests/pre_totalseg.jsonl \
  --inspect-labels --allow-missing-roi --context-k 3 --seed 20260831

MEDVERSE_PROJECT_DIR=$PWD \
SOURCE_MANIFEST=$PWD/work/paot2_icl_inspected.jsonl \
TOTALSEG_PYTHON=$(which python) TOTALSEG_CLI=$(which TotalSegmentator) \
MAX_CASES_PER_STRATUM=1 sbatch slurm/run_totalsegmentator_organs.sbatch
```

smoke 成功并人工检查 `work/data/derived_liver_kidney_v1/qc` 后，移除
`MAX_CASES_PER_STRATUM` 执行全量器官定位。脚本幂等：已完成且几何校验通过
的病例会跳过；失败、空 mask 和异常体积写入 JSONL/summary，不静默丢弃。
原始 NIfTI、标签、器官 mask 和患者目录不得提交 Git。

## 阶段 2：共享 ROI 与 nnU-Net planner

```bash
MEDVERSE_PROJECT_DIR=$PWD \
SOURCE_MANIFEST=$PWD/work/paot2_icl_inspected.jsonl \
DATA_PYTHON=$(which python) NNUNET_BIN=$(which nnUNetv2_plan_and_preprocess) \
sbatch slurm/prepare_shared_roi_dataset.sbatch
```

产物位于 `work/data/derived_liver_kidney_v1`，含二值肿瘤 mask、器官 mask、
ROI、几何元数据、哈希、体积/直径和三平面 QC。两组模型必须读取同一个
`roi_manifest.jsonl` 和 `nnunet_plan_snapshot.json`。

## 阶段 3：模型 smoke 与 BAM 机制证明

```bash
MEDVERSE_PROJECT_DIR=$PWD MEDVERSE_PYTHON=$(which python) \
sbatch slurm/smoke_liver_kidney_models.sbatch
```

smoke 同时检查 No-ICL nnU-Net 构件和 ICL 模型的一次前向、反向、权重
保存/加载。ICL 检查 K=1/K=3、空 context mask、CCTI 模块为空，并通过
forward hook 记录原始 BAM cross-attention 确实收到 context feature；替换
context 后输出必须发生非零变化。

## 阶段 4：2–4 病例过拟合门禁

K=3 需要每任务至少 4 个不同患者：

```bash
python scripts/prepare_overfit_manifest.py \
  --manifest work/data/derived_liver_kidney_v1/manifests/roi_manifest.jsonl \
  --output work/experiments/overfit/manifest.jsonl \
  --cases-per-task 4 --context-k 3
```

分别在 GPU 0/GPU 1 训练并在同一 8 例上预测。统一评价后，只有平均 Dice
至少 0.80 且单例最低 Dice 至少 0.60 才创建门禁：

```bash
python scripts/check_overfit_gate.py --metrics-csv <NO_ICL_CSV> \
  --gate work/experiments/gates/no_icl_overfit.pass
python scripts/check_overfit_gate.py --metrics-csv <ICL_CSV> \
  --gate work/experiments/gates/icl_bam_k3_overfit.pass
```

若不能过拟合，停止正式实验，先查标签、几何、loss、预训练迁移和 BAM
context 路径，不允许用环境变量跳过门禁开展正式结果。

## 阶段 5：双 GPU 正式训练

```bash
MEDVERSE_PROJECT_DIR=$PWD \
MEDVERSE_PYTHON=$(which python) \
sbatch slurm/run_two_gpu_main_experiment.sbatch
```

`scripts/run_two_gpu_main_experiment.sh` 自动检测两张可见 GPU：GPU 0 只启动
No-ICL，GPU 1 只启动 ICL+BAM K3；日志、PID、配置快照和 checkpoint 使用
不同目录。可单独恢复：

```bash
NO_ICL_MODE=resume bash scripts/run_two_gpu_main_experiment.sh no-icl
ICL_MODE=resume bash scripts/run_two_gpu_main_experiment.sh icl
```

脚本不启动第三个实验。显存不足时依次使用 AMP、batch=1、梯度累积、K=3
顺序/分块编码、activation checkpointing，最后才减 patch；不得改变病例、
ROI 物理范围、spacing、标签或测试方式。

## Context 消融

主实验先使用 fixed `random_same_task`。构建器支持 K=1/3/5，val/test 也只
从 train bank 取非空完整标注且不同患者的 context：

```bash
python scripts/build_liver_kidney_contexts.py --manifest <ROI_MANIFEST> \
  --output <RANDOM_JSONL> --strategy random_same_task --k 3
python scripts/build_liver_kidney_contexts.py --manifest <ROI_MANIFEST> \
  --output <RETRIEVAL_JSONL> --strategy retrieval_same_task --k 3 \
  --embedding-dir work/context_embeddings
python scripts/build_liver_kidney_contexts.py --manifest <ROI_MANIFEST> \
  --output <WRONG_JSONL> --strategy wrong_task --k 3
```

retrieval embedding 是冻结的 CT+器官 ROI 输入特征，不读 target mask，并
保存 Top-20 候选和相似度。wrong-task 只做机制诊断。主模型结果不支持 ICL
时，才按 wrong/random/retrieval/BAM-off/K1/K5 顺序，每批最多两个消融。

## 原空间推理、评价与报告

Medverse 推理会先生成 ROI mask，再按保存的逆变换还原：

```bash
python scripts/infer_pan_cancer_icl.py --manifest <ROI_MANIFEST> \
  --checkpoint <BEST_PT> --target-spacing 1.2,1.2,2.5 \
  --output-dir work/predictions/icl
```

nnU-Net 使用 `nnUNetv2_predict` 推理 Dataset501 的 `imagesTs`，随后执行：

```bash
python scripts/restore_nnunet_predictions.py --manifest <ROI_MANIFEST> \
  --roi-predictions <NNUNET_ROI_PRED_DIR> --output-dir work/predictions/no_icl
```

统一评价严格校验 prediction/GT 的原始 shape 和 affine，输出逐例 CSV、JSON
及论文 Markdown 表格：

```bash
python scripts/evaluate_liver_kidney_original_space.py \
  --manifest <ROI_MANIFEST> \
  --predictions work/predictions/icl/predictions.jsonl \
  --comparison-predictions work/predictions/no_icl/predictions.jsonl \
  --output-dir work/reports/icl_vs_no_icl --nsd-tolerance-mm 2
```

指标包括 Dice、NSD、HD95、lesion precision/recall/F1、FP lesion/case，按
任务、数据源、最大病灶直径 `<20 mm`/`>=20 mm` 分层，并报告 patient macro、
organ macro、逐例差值、paired bootstrap 95% CI 和 Wilcoxon。测试集仅在
模型和阈值冻结后执行一次。

## 输出与审计

正式目录固定为：

```text
work/experiments/
  no_icl_nnunet/{config.yaml,logs/,checkpoints/}
  icl_medverse_bam_k3/{config.yaml,logs/,checkpoints/}
```

日志需保留 git commit、配置快照、软件版本、随机种子、GPU/显存、参数量、
更新次数、训练/推理时间与 context 额外开销。只有 ICL 在肝或肾至少一项稳定
改善、小病灶 recall 明显改善，或总体接近但跨数据集更稳定，才继续诊断消融
和最终 5-fold；5-fold 仍按 GPU 0 baseline、GPU 1 ICL 的单进程队列运行。
