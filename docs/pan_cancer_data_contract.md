# Pan-cancer binary segmentation data contract

每行 JSONL 表示一个“患者—检查—序列—标注目标”，必须包含：

```json
{"case_id":"C001_CT_PV_primary","patient_id":"C001","study_id":"S001","image":"data/C001/image.nii.gz","mask":"data/C001/primary_tumor.nii.gz","split":"train","cancer_type":"PDAC","primary_organ":"pancreas","target_region":"primary_tumor","modality":"CT","phase_or_sequence":"portal_venous","center":"center_a","annotation_protocol":"v1"}
```

约束：

- `split` 只能是 `train`、`val` 或 `test`，同一患者不能跨 split。
- `image` 与 `mask` 必须是工程根目录相对路径或绝对路径。
- context 与 target 必须有相同的 `target_region`、`modality` 和 `primary_organ`；
  context mask 是该二值任务的前景定义。
- 只有 train 条目可进入 context bank；target 和 context 不能来自同一患者。
- 脑肿瘤 WT、TC、ET 等目标必须写成不同 `target_region`，不得混库。

第一轮 CCTI 消融应离线固定每个 target 的 context IDs，确保四个实验臂看到完全
相同的病例。待通道选择机制验证后，再加入 embedding 检索。
