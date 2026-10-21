# 5090 远程执行约定

更新：2026-10-21。用户已指定：本地设计和编辑，后续执行均在 5090。

## 已确认

- SSH 别名：`5090`；现有密钥和主机校验可正常连接。
- 系统：Windows；主机名 `DESKTOP-OD6G5KR`。
- GPU：RTX 5090，显存 32,607 MiB。检查时使用约 138 MiB、利用率 0%，未见训练进程；启动前再次检查。
- Git、CUDA 工具、Hugging Face CLI 已存在；HF 当前未登录。
- GitHub 应用已连接到 `HKUJasonJiang`；本地 experiment 尚无远程仓库。
- 环境与数据检查尚未在服务器运行；以上为只读连接和软件清单检查。

## 存储分工

| 位置 | 内容 |
| --- | --- |
| GitHub 私有代码仓库 | experiment 源码、配置、OKR、设计说明、小型图表和指标 |
| HF 私有 Dataset 仓库 | 地震动、结构清单、仿真 HDF5、训练数据分片、划分清单及哈希 |
| HF Model 仓库（首次权重备份前建立） | 模型权重、归一化尺度、模型配置、对应代码与数据版本 |
| 5090 项目目录 | 计算环境、数据缓存、运行日志、中间结果和活动训练任务 |

每次运行关联 Git commit、数据仓库 revision、配置和 seed。
服务器生成的新数据直接上传 HF；本地主要取回图表和摘要。
MDOF 求解器及旧参数生成器依赖仍需随服务器部署一并准备，不能只上传 Transformer 文件后假定数据生成器已就绪。

## 待提供的信息

1. GitHub 私有空仓库 URL，建议名 `seismic-transformer-experiments`；现有本地仓库已有历史，创建时不初始化 README。
2. HF 私有 Dataset 仓库 URL，建议名 `seismic-transformer-stage1`。
3. 服务器对上述仓库的访问认证已配置的确认。不要在聊天中提交 token。

GitHub 可以使用授权给当前连接的仓库访问、服务器 SSH key 或仅针对该仓库的 fine-grained Contents 读写 token。
HF 使用仅覆盖相关仓库读写的 fine-grained token，在服务器执行 `hf auth login` 登录或由服务器进程读取 `HF_TOKEN`。
凭据不进入配置、源码、日志或 Git 远程 URL。

## 下一次执行

1. 连接并检查 5090 GPU；设置独立项目环境和目录。
2. 同步精确代码版本、MDOF 依赖及预览数据，记录来源哈希。
3. 在服务器运行 patch=60 的 O2 检查，保存 `reports/model_check_patch60.json`。
4. 通过后建立小样本训练入口，并在服务器开展 O3。

当前 patch=60 已写入配置，尚未在服务器验证；历史 patch=10 的检查结果仍保留，不能当成 patch=60 的验证。
