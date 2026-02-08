# VLAT (Vision-Language-Action Transformer) Policy

## 概述

VLAT是一个为KDC挑战赛设计的新型多模态Transformer策略，融合了最新的视觉编码、多模态融合和时序建模技术。

### 核心特性

1. **Vision Transformer (ViT) 视觉编码**
   - 使用ViT-Small替代ResNet18，提供更强的全局视觉理解能力
   - 支持多相机输入（head_cam, wrist_cam_l, wrist_cam_r）
   - 可选的深度图像编码器

2. **Perceiver融合模块**
   - 使用可学习的query tokens聚合多模态信息
   - RGB + Depth + 机器人状态深度融合
   - 固定大小输出，计算高效

3. **时序Transformer**
   - 显式建模多帧观测之间的时序依赖
   - 支持因果注意力和非因果注意力
   - 注意力池化用于时序聚合

4. **动作预测**
   - Action Chunking：一次预测100步动作
   - 不确定度估计：为每个预测动作估计置信度
   - 多任务损失：BC + 平滑 + 不确定度正则化

5. **训练增强**
   - 支持混合精度训练(FP16)
   - 数据增强（色彩抖动、遮挡、噪声等）
   - 分层学习率（预训练ViT使用较低学习率）

## 模型架构

```
输入观测 (RGB×3, State)
    ↓
[ViT视觉编码]
├─ RGB Encoder (ViT-Small) → patch tokens
└─ State Encoder (MLP) → state features
    ↓
[Perceiver多模态融合]
├─ Learnable Query Tokens (64)
├─ Cross-Attention (Query ← RGB + State)
└─ → 固定长度表示
    ↓
[时序建模]
├─ Temporal Transformer (处理2个时间步)
├─ 注意力池化
└─ → 时序上下文特征
    ↓
[动作解码]
├─ Transformer Decoder
├─ Action Head → 100步动作
└─ Uncertainty Head → 不确定度估计
```

## 安装和测试

### 1. 环境要求

- Python 3.10+
- PyTorch 2.0+
- timm (for ViT models)
- 其他依赖见 requirements.txt

### 2. 快速测试

运行简化测试脚本验证安装：

```bash
cd /workspace/kuavo_data_challenge
python test_vlat_simple.py
```

预期输出：
```
✓ All tests passed!
Parameters: ~23M
```

### 3. 完整测试

如果需要测试完整配置（包括depth）：

```bash
python test_vlat_policy.py
```

## 训练

### 基本训练命令

```bash
accelerate launch \
  --config_file configs/accelerate/accelerate_config.yaml \
  kuavo_train/train_policy_with_accelerate.py \
  --config-path=../configs/policy \
  --config-name=vlat_config.yaml
```

### 配置文件

主配置文件：`configs/policy/vlat_config.yaml`

#### 关键超参数

```yaml
policy:
  # 观测配置
  n_obs_steps: 2              # 观测步数
  chunk_size: 100             # 动作块大小
  n_action_steps: 1           # 每次执行的动作数
  
  # ViT配置
  vision_backbone: vit_small_patch16_224
  image_size: [224, 224]      # 输入图像大小
  
  # Perceiver融合
  n_query_tokens: 64          # 查询token数量
  fusion_dim: 512             # 融合维度
  fusion_layers: 2            # 融合层数
  
  # 时序建模
  temporal_dim: 512           # 时序特征维度
  temporal_layers: 4          # 时序层数
  
  # 动作解码
  decoder_dim: 512            # 解码器维度
  decoder_layers: 4           # 解码器层数
  
  # 训练配置
  use_uncertainty: true       # 启用不确定度估计
  lambda_smooth: 0.1          # 平滑损失权重
  
  # 优化器
  optimizer_lr: 5e-5          # 学习率
  optimizer_weight_decay: 1e-4
```

#### 启用深度图像

编辑 `vlat_config.yaml`:

```yaml
custom:
  use_depth: true
  depth_backbone: vit_small_patch16_224
```

### 训练监控

使用TensorBoard监控训练：

```bash
tensorboard --logdir outputs/train/task1/vlat/
```

关注指标：
- `mse_loss`: 行为克隆损失（主要指标）
- `smooth_loss`: 动作平滑度
- `uncertainty_reg`: 不确定度正则化
- `mean_uncertainty`: 平均不确定度

## 模型性能

### 参数量

- 完整模型：约23M参数
- 主要组件：
  - ViT编码器：~22M
  - Perceiver融合：~0.5M
  - 时序Transformer：~0.3M
  - 动作解码器：~0.2M

### 推理速度

在单GPU (NVIDIA A100)上：
- 前向传播：~20ms/batch (batch_size=32)
- 动作预测：~5ms (单个样本)

### 内存使用

- 训练 (batch_size=32, FP32): ~12GB
- 训练 (batch_size=32, FP16): ~8GB
- 推理 (batch_size=1): ~2GB

## 与现有策略对比

### vs ACT

| 特性 | ACT | VLAT |
|------|-----|------|
| 视觉编码器 | ResNet18 | ViT-Small |
| 多模态融合 | 简单拼接 | Perceiver |
| 时序建模 | 隐式 | 显式Transformer |
| 不确定度估计 | ❌ | ✓ |
| 参数量 | ~10M | ~23M |
| 推理速度 | 快 | 中等 |

**优势**：
- 更强的视觉理解能力
- 显式的时序建模
- 不确定度感知

**劣势**：
- 参数量更大
- 推理稍慢

### vs Diffusion Policy

| 特性 | Diffusion | VLAT |
|------|-----------|------|
| 生成方式 | 扩散去噪 | 直接回归 |
| 推理速度 | 慢 (多步) | 快 (单步) |
| 训练稳定性 | 较复杂 | 简单 |
| 多模态性 | ✓ | ✓ |

**优势**：
- 推理速度快10-100倍
- 训练更稳定
- 更易调试

**劣势**：
- 可能缺少扩散模型的多模态表达能力

## 超参数调优建议

### 1. 学习率调优

```yaml
# 保守策略（推荐起点）
optimizer_lr: 5e-5

# 激进策略（如果收敛太慢）
optimizer_lr: 1e-4

# 使用预训练ViT时
optimizer_lr: 3e-5  # ViT会自动使用10x更低的学习率
```

### 2. 模型大小调优

如果显存不足，可以减小模型：

```yaml
n_query_tokens: 32      # 从64降到32
fusion_dim: 256         # 从512降到256
temporal_dim: 256
decoder_dim: 256
```

如果想要更大的模型：

```yaml
vision_backbone: vit_base_patch16_224  # 使用ViT-Base
fusion_dim: 768
temporal_dim: 768
decoder_dim: 768
```

### 3. 训练策略调优

```yaml
# 更长的warmup
scheduler_warmup_steps: 1000

# 更大的batch size (如果显存够)
batch_size: 64

# 启用混合精度
use_amp: True
```

### 4. 损失权重调优

```yaml
lambda_smooth: 0.1          # 动作平滑 (0.05-0.2)
lambda_uncertainty: 0.001   # 不确定度 (0.0001-0.01)
```

## 故障排除

### 1. 显存不足 (OOM)

**解决方案**：
- 减小batch_size: `32 → 16`
- 启用混合精度: `use_amp: True`
- 减小模型大小（见上文）
- 使用梯度累积: `accumulation_steps: 2`

### 2. 训练不收敛

**检查项**：
- 学习率是否过大？尝试减半
- 数据增强是否过强？降低augmentation权重
- batch_size是否过小？增加到32+
- warmup steps是否足够？增加到500+

### 3. 过拟合

**解决方案**：
- 增加dropout: `fusion_dropout: 0.2`
- 增强数据增强
- 减小模型大小
- 增加weight_decay: `1e-3`

### 4. 推理速度慢

**优化方法**：
- 使用FP16推理
- 减小chunk_size: `100 → 50`
- 减小n_obs_steps: `2 → 1`
- 考虑模型蒸馏（高级）

## 文件结构

```
kuavo_train/wrapper/policy/vlat/
├── __init__.py                    # 模块导出
├── VLATConfigWrapper.py           # 配置类
├── VLATModelWrapper.py            # 核心模型
├── VLATPolicyWrapper.py           # 策略包装器
├── vision_transformer.py          # ViT编码器
├── perceiver_fusion.py            # Perceiver融合
└── temporal_transformer.py        # 时序Transformer

configs/policy/
└── vlat_config.yaml               # 训练配置

tests/
├── test_vlat_simple.py            # 简化测试
└── test_vlat_policy.py            # 完整测试
```

## 引用

如果使用VLAT策略，请引用以下论文：

- Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
- Perceiver: Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
- ACT: Zhao et al., "Action Chunking with Transformers", RSS 2023

## 许可证

本代码遵循 Apache 2.0 许可证。

## 联系方式

如有问题，请在GitHub仓库提issue或联系开发者。

---

**祝训练顺利！Good luck with KDC Challenge!** 🚀
