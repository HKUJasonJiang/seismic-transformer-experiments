# 实验版本

每个想法独立保存代码、配置、设计和结果，不覆盖旧版本。代码与简明记录进 Git；完整结果、图和权重保存在同目录的私有归档，数据按哈希共用。

| 版本 | 想法 | 记录 |
|---|---|---|
| exp1-joint-basis | 已有 A+B 联合训练及空间基改进 | [exp1.md](exp1-joint-basis/exp1.md) |
| exp2-module-isolation | 固定 A/B、独立顶层与结构留出 | [exp2.md](exp2-module-isolation/exp2.md) |
| exp3-response-teacher | 已完成：C+B 可重建，冻结后训练 A 未稳定改善留出预测 | [exp3.md](exp3-response-teacher/exp3.md) |
| exp4-unseen-earthquakes | 已完成：固定 A+B，已见结构遇到未训练地震动，非线性损伤明显低估 | [exp4.md](exp4-unseen-earthquakes/exp4.md) |
| exp5-normalized-floor-query | 共享归一化楼层查询；多结构、多地震动训练，周期和地震动分别测试 | [exp5.md](exp5-normalized-floor-query/exp5.md) |
| exp6-gm-scaling-structure-ood | 已合并到 exp7，只保留原始设计 | [exp6.md](exp6-gm-scaling-structure-ood/exp6.md) |
| exp7-cross-attention-decoder | 已完成：KNET 框架数据集（58,424 算例）、训练记录数缩放曲线、层数与 T1 内插外推；exp5 风格 MLP 解码器优于交叉注意力，加地面项改善 5–10 Hz 但 10 Hz 以上多出能量 | [exp7.md](exp7-cross-attention-decoder/exp7.md) |
| exp8-gm-encoder-query-decoders | 运行中：编码器只读地震动、建筑信息只进解码器（可一次编码多次查询）；解码器为带结构 FiLM 的 MLP 或全频带 FNO，与 exp7 最佳模型对照 | [exp8.md](exp8-gm-encoder-query-decoders/exp8.md) |

历史两个版本保留原报告及原始指标，不能把旧指标改名成 MSE。新版本面向用户统一报告物理单位 MSE。历史源文件按 `codes/src`、`codes/scripts`、`codes/configs` 保留原相对布局；复现需要配置同版数据与权重目录，实际运行来源以原 run 的 provenance 为准。
