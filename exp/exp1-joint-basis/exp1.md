# exp1 — 联合 A+B 与空间基查询

状态：历史实验归档。原 stage1 文件保留不变，代码副本位于 codes，复制来源与文件哈希见 configs/source_snapshot.json。

## 设计与配置

A 为 4 层 Transformer，patch=60，d=128，FiLM 接收结构参数。B 用 8 组时间系数头和确定的离散正弦楼层基，支持 3/5/8 层逐层查询。基函数不是识别出的物理模态，不是时间 FFT。

主要版本使用已有 65 个 DS>2 训练算例，联合监督全楼位移、加速度和漂移。原配置在 codes/configs/diagnosis.json，原代码包括八组拟合诊断，不把自由 latent 对照作为预测。

## 结果

训练拟合通过，冻结后开发验证记录上的泛化未通过。当前物理输入推理权重为原 `stage1/runs/diagnosis-nonlinear-v1-basis_joint/inference.pt`。完整历史报告复制到 results，主要时程复制到 figures；权重与数据仍按原哈希引用，原始运行 commit 不以归档 commit 替代。

原记录：[诊断安排](../../stage1/DIAGNOSIS_PLAN.md)。后续归因以 exp2 的固定权重干预为准。
