# Seismic Transformer Experiments

独立实验仓库。代码、配置、实验安排和记录均放在本目录；不修改论文其他目录。

## 当前阶段

[Stage 1 — Proof of Concept](stage1/OKR.md)：先验证模型能否从少量建筑属性和地震动学会逐层响应。

[当前模型设计](stage1/MODEL_DESIGN.md)：FiLM 时序编码器 + 共享楼层查询解码器，联合输出逐层位移与绝对加速度。

[远程执行约定](stage1/REMOTE_EXECUTION.md)：本地设计和编辑；所有仿真、预处理、测试、训练和评估在 5090 执行。代码已同步公开 GitHub，数据和权重保存在私有 HF。

## 目录

```text
stage1/
  OKR.md       实验目标、关键结果与状态
  src/         模型、数据处理与指标代码
  scripts/     数据生成、训练与评估入口
  configs/     可复现实验配置
  data/        数据与划分清单（不提交 Git）
  runs/        权重、日志与预测（不提交 Git）
  reports/     结果摘要、小型指标表与图
```

## 管理约定

- Git 仓库根目录为 `experiment/`；公开远程仓库已配置。
- 每个训练任务使用唯一 run ID，记录 Git commit、配置、随机种子和数据清单哈希；有未提交代码时一并保存差异。
- 只提交代码、配置、Markdown 和小型结果；不提交数据集、权重、环境或缓存。
- 新 MDOF 工作台作为只读依赖；数据生成入口和所有新输出保存在本仓库。记录求解器版本与参数生成方式。
- 状态只按实际结果更新；已完成[五层结构 T1/PGA 预览](stage1/reports/five_story_preview/README.md)、[patch=60 服务器核验](stage1/reports/model_check_patch60.json)和 24 个单波训练算例的拟合。
- 后续 [PEER-10 试验](stage1/PEER10_PLAN.md)已完成 3/5/8 层联合训练、非线性覆盖与独立记录测试：查询接口通过，预测精度未达到泛化门槛。完整数据、权重、逐算例指标和响应图保存在私有存储及本地 `stage1/reports/`；整个 Stage 1 仍为 PARTIAL。
- [A/B 训练诊断](stage1/DIAGNOSIS_PLAN.md)使用原 65 个非线性训练样本，保留 A 并改进 B 后，3/5/8 层分别通过训练拟合门槛。冻结后的原验证波快照仍未通过泛化门槛，未据此继续调参。未扩大数据；八组对照、推理权重和时程图均已私有归档。
