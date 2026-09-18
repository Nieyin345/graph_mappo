# 数据生成工具

这里存放训练前的物理数据生成与维护脚本，不属于强化学习核心实现。

## 主要工具

- `build_global_tensor.py`：生成全年卫星/HAP/链路 QKD 速率数据。
- `fetch_weather.py`：下载 GS 气象数据。
- `zero_rate_link_rebuild.py`：清理全年零速率链路并重建 HDF5。
- `legacy_data.py`：旧版 GS/HAP/SAT 数据模型及全球节点数据库，仅供数据生成脚本使用。
- `legacy_physics.py`：旧版几何、HAP 运动和轨道传播工具，仅供数据生成脚本使用。

## 运行原则

项目运行路径统一以仓库根目录为基准；脚本内部使用自身位置推导项目根目录，避免依赖当前 shell 的工作目录。

生成的数据写入 `dataset/global/`，天气原始数据写入 `weather/2023/`。
