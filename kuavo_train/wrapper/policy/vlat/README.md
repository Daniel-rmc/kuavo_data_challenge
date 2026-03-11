# VLAT Policy（Vision-Language-Action Transformer）

本目录实现了 KDC 项目中的 VLAT 策略，目标是在双臂任务上统一建模视觉、状态与时序信息，输出动作序列（action chunk）并可选预测不确定度。

## 1. 目录与实现对应

- `VLATConfigWrapper.py`
  - `CustomVLATConfigWrapper`，注册名为 `custom_vlat`
  - 定义模型超参数、优化器/调度器、特征校验与配置保存加载
- `vision_transformer.py`
  - `VisionTransformerEncoder`：基于 timm 的 ViT 编码器
  - `RGBViTEncoder`：多相机 RGB 编码（共享或独立编码器）
  - `DepthViTEncoder`：深度输入编码（支持 1 通道）
- `perceiver_fusion.py`
  - Perceiver 多模态融合模块（RGB/Depth/State）
- `temporal_transformer.py`
  - 时序建模模块，支持因果/非因果注意力
- `VLATModelWrapper.py`
  - `VLATModel`：组合视觉编码、融合、时序编码、动作解码、不确定度头
- `VLATPolicyWrapper.py`
  - `CustomVLATPolicyWrapper`：训练前向、损失计算、动作分块推理 `select_action`

模块导出在 `__init__.py`：
- `CustomVLATConfigWrapper`
- `VLATModel`
- `CustomVLATPolicyWrapper`

## 2. 模型与数据流

输入（训练时）主要包含：
- 多路图像：`observation.images.*`
- 机器人状态：`observation.state`
- 可选深度：`observation.depth.*`

主流程：
1. RGB/Depth 通过 ViT 提取 patch tokens
2. 可选 RGB-Depth cross-attention
3. 与状态向量在 Perceiver 中融合为固定长度 query tokens
4. 经过 Temporal Transformer 聚合多观测步
5. Transformer decoder 输出 `chunk_size x action_dim` 动作序列
6. 可选不确定度头输出逐动作维度的不确定度

策略层损失（`VLATPolicyWrapper.forward`）包含：
- 行为克隆 MSE
- 平滑损失（动作相邻步差分约束）
- 不确定度正则与加权误差项（可选）

## 3. 关键配置文件

训练主配置：`configs/policy/vlat_config.yaml`

重点字段：
- 基本：`policy_name: vlat`
- 数据：
  - `root: /workspace/datasets/ICRA/lerobotdata/task1`
  - `repoid: lerobot/task1`（本地元数据缺失时才会触发 Hub 查询）
- 视觉：`vision_backbone: vit_small_patch16_224`
- 时序/解码：`temporal_dim`, `decoder_dim`, `chunk_size`
- 自定义项：
  - `custom.use_depth`
  - `custom.use_separate_rgb_encoder_per_camera`
  - `custom.use_separate_depth_encoder_per_camera`

Accelerate 配置：`configs/accelerate/accelerate_config.yaml`
- 当前按 2 卡环境配置：
  - `num_processes: 2`
  - `gpu_ids: "0,1"`

## 4. 启动前检查

在项目根目录执行：

```bash
cd /workspace/kuavo_data_challenge

# 1) 基础单测
python test_vlat_policy.py

# 2) 可见 GPU
python - <<'PY'
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

建议保证数据目录存在以下文件：
- `/workspace/datasets/ICRA/lerobotdata/task1/meta/info.json`
- `/workspace/datasets/ICRA/lerobotdata/task1/meta/stats.json`

## 5. 启动训练

### 5.1 多卡训练（推荐）

```bash
cd /workspace/kuavo_data_challenge
accelerate launch \
  --config_file configs/accelerate/accelerate_config.yaml \
  kuavo_train/train_policy_with_accelerate.py \
  --config-path=../configs/policy \
  --config-name=vlat_config.yaml
```

### 5.2 单卡调试（更稳定）

```bash
cd /workspace/kuavo_data_challenge
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  kuavo_train/train_policy_with_accelerate.py \
  --config-path=../configs/policy \
  --config-name=vlat_config.yaml \
  training.num_workers=0
```

### 5.3 TensorBoard 监控

```bash
tensorboard --logdir /workspace/kuavo_data_challenge/outputs/train/task1/vlat
```

## 6. 常见问题排查

### 问题 A：invalid device ordinal
表现：`CUDA error: invalid device ordinal`

原因：`accelerate_config.yaml` 中的 `num_processes` 或 `gpu_ids` 超出实际可见 GPU 数。

处理：
- 先检查 `torch.cuda.device_count()`
- 将 `num_processes` 与 `gpu_ids` 调整到一致

### 问题 B：Error locating target ... CustomVLATConfigWrapper
表现：Hydra 无法定位 `kuavo_train.wrapper.policy.vlat...`

原因：运行时导入路径未指向当前工程源码，或环境中存在旧版本包。

处理：
- 在项目根执行命令
- 优先使用本仓库的训练脚本与配置
- 必要时显式设置：

```bash
export PYTHONPATH=/workspace/kuavo_data_challenge:$PYTHONPATH
```

### 问题 C：Processor for policy type 'custom_vlat' is not implemented
原因：`lerobot` 的 processor factory 未为 `custom_vlat` 注册分支。

处理：
- 在 `third_party/lerobot/src/lerobot/policies/factory.py` 中添加 `custom_vlat` 分支（通常可复用 ACT 的 pre/post processor 流程）

### 问题 D：本地数据路径不存在触发 Hub 404
表现：`RepositoryNotFoundError: ... /api/datasets/lerobot/task1/refs`

原因：本地 `root` 目录没有 metadata，框架回退尝试访问 Hub 仓库。

处理：
- 确认 `vlat_config.yaml` 的 `root` 指向真实本地数据目录
- 确认 `meta/info.json` 与 `meta/stats.json` 存在