# 实验版本

每个想法独立保存代码、配置、设计和结果，不覆盖旧版本。代码与简明记录进 Git；完整结果、图和权重保存在同目录的私有归档，数据按哈希共用。

| 版本 | 想法 | 记录 |
|---|---|---|
| exp1-joint-basis | 已有 A+B 联合训练及空间基改进 | [exp1.md](exp1-joint-basis/exp1.md) |
| exp2-module-isolation | 固定 A/B、独立顶层与结构留出 | [exp2.md](exp2-module-isolation/exp2.md) |
| exp3-response-teacher | 已完成：C+B 可重建，冻结后训练 A 未稳定改善留出预测 | [exp3.md](exp3-response-teacher/exp3.md) |
| exp4-unseen-earthquakes | 已完成：固定 A+B，已见结构遇到未训练地震动，非线性损伤明显低估 | [exp4.md](exp4-unseen-earthquakes/exp4.md) |
| exp5-normalized-floor-query | 共享归一化楼层查询；多结构、多地震动训练，周期和地震动分别测试 | [exp5.md](exp5-normalized-floor-query/exp5.md) |
| exp6-gm-scaling-structure-ood | 设计中：先只增加训练地震动条数（模型同 exp5）看新地震动误差能否下降；再让刚度、强度独立变化并留出层数，测试新结构 | [exp6.md](exp6-gm-scaling-structure-ood/exp6.md) |
| exp7-cross-attention-decoder | 代码已写、未运行：解码器改为交叉注意力，输出相对加速度，查询带局部地震动片段，幅值按 PGA 分解；先在 exp5 数据上五个变体逐步对照 | [exp7.md](exp7-cross-attention-decoder/exp7.md) |

历史两个版本保留原报告及原始指标，不能把旧指标改名成 MSE。新版本面向用户统一报告物理单位 MSE。历史源文件按 `codes/src`、`codes/scripts`、`codes/configs` 保留原相对布局；复现需要配置同版数据与权重目录，实际运行来源以原 run 的 provenance 为准。
