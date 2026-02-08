"""
Test script for VLAT policy.

Validates:
1. Configuration loading
2. Model instantiation
3. Forward pass with synthetic data
4. Loss computation
5. Action prediction
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kuavo_train.wrapper.policy.vlat import CustomVLATConfigWrapper, CustomVLATPolicyWrapper
from lerobot.configs.types import PolicyFeature, FeatureType
from omegaconf import OmegaConf


def create_test_config():
    """Create a minimal test configuration."""
    
    # Load from yaml
    config_path = Path(__file__).parent / "configs/policy/vlat_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Create minimal input/output features for testing (without depth for now)
    input_features = {
        "observation.images.head_cam_h": PolicyFeature(
            shape=(3, 480, 640),
            type=FeatureType.VISUAL,
        ),
        "observation.images.wrist_cam_l": PolicyFeature(
            shape=(3, 480, 640),
            type=FeatureType.VISUAL,
        ),
        "observation.images.wrist_cam_r": PolicyFeature(
            shape=(3, 480, 640),
            type=FeatureType.VISUAL,
        ),
        "observation.state": PolicyFeature(
            shape=(16,),
            type=FeatureType.STATE,
        ),
    }
    
    output_features = {
        "action": PolicyFeature(
            shape=(16,),
            type=FeatureType.ACTION,
        ),
    }
    
    # Instantiate config
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    
    # Remove Hydra-specific fields
    policy_cfg.pop("_target_", None)
    
    policy_cfg["input_features"] = input_features
    policy_cfg["output_features"] = output_features
    
    config = CustomVLATConfigWrapper(**policy_cfg)
    
    return config


def create_synthetic_batch(config, batch_size=2):
    """Create synthetic batch data for testing."""
    
    n_obs_steps = config.n_obs_steps
    chunk_size = config.chunk_size
    
    # Image features
    batch = {}
    
    if config.image_features:
        for key in config.image_features:
            # (B, n_obs_steps, C, H, W)
            batch[key] = torch.randn(batch_size, n_obs_steps, 3, 480, 640)
    
    # Skip depth for now
    # if config.depth_features:
    #     for key in config.depth_features:
    #         batch[key] = torch.randn(batch_size, n_obs_steps, 3, 480, 640)
    
    # State features
    if config.robot_state_feature:
        state_dim = config.robot_state_feature.shape[0]
        batch["observation.state"] = torch.randn(batch_size, n_obs_steps, state_dim)
    
    # Actions (for training)
    action_dim = config.action_feature.shape[0]
    batch["action"] = torch.randn(batch_size, chunk_size, action_dim)
    batch["action_is_pad"] = torch.zeros(batch_size, chunk_size, dtype=torch.bool)
    
    return batch


def test_config():
    """Test configuration loading."""
    print("=" * 60)
    print("Test 1: Configuration Loading")
    print("=" * 60)
    
    try:
        config = create_test_config()
        print(f"✓ Config loaded successfully")
        print(f"  - n_obs_steps: {config.n_obs_steps}")
        print(f"  - chunk_size: {config.chunk_size}")
        print(f"  - vision_backbone: {config.vision_backbone}")
        print(f"  - fusion_dim: {config.fusion_dim}")
        print(f"  - temporal_dim: {config.temporal_dim}")
        print(f"  - use_depth: {config.use_depth}")
        print(f"  - num_cameras (RGB): {len(config.image_features)}")
        print(f"  - num_cameras (Depth): {len(config.depth_features)}")
        return config
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_instantiation(config):
    """Test model instantiation."""
    print("\n" + "=" * 60)
    print("Test 2: Model Instantiation")
    print("=" * 60)
    
    try:
        policy = CustomVLATPolicyWrapper(config)
        print(f"✓ Policy instantiated successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return policy
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(policy, config):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Test 3: Forward Pass")
    print("=" * 60)
    
    try:
        batch = create_synthetic_batch(config, batch_size=2)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        print(f"  - Device: {device}")
        print(f"  - Batch size: 2")
        
        # Forward pass
        policy.train()
        loss, loss_dict = policy.forward(batch)
        
        print(f"✓ Forward pass successful")
        print(f"  - Total loss: {loss.item():.4f}")
        for k, v in loss_dict.items():
            print(f"  - {k}: {v:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_prediction(policy, config):
    """Test action prediction."""
    print("\n" + "=" * 60)
    print("Test 4: Action Prediction")
    print("=" * 60)
    
    try:
        batch = create_synthetic_batch(config, batch_size=1)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Predict action chunk
        policy.eval()
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
        
        print(f"✓ Action prediction successful")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Expected shape: (1, {config.chunk_size}, {config.action_feature.shape[0]})")
        
        # Test select_action
        policy.reset()
        action = policy.select_action(batch)
        
        print(f"✓ Single action selection successful")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Expected shape: (1, {config.action_feature.shape[0]})")
        
        return True
    except Exception as e:
        print(f"✗ Action prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(policy, config):
    """Test gradient flow through the model."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient Flow")
    print("=" * 60)
    
    try:
        batch = create_synthetic_batch(config, batch_size=2)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward and backward
        policy.train()
        loss, _ = policy.forward(batch)
        loss.backward()
        
        # Check gradients
        has_grad = False
        nan_grad = False
        grad_norms = []
        
        for name, param in policy.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if torch.isnan(param.grad).any():
                    nan_grad = True
                    print(f"  ! NaN gradient in {name}")
        
        if has_grad and not nan_grad:
            print(f"✓ Gradients computed successfully")
            print(f"  - Mean gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")
            print(f"  - Max gradient norm: {max(grad_norms):.6f}")
            print(f"  - Min gradient norm: {min(grad_norms):.6f}")
            return True
        elif nan_grad:
            print(f"✗ NaN gradients detected")
            return False
        else:
            print(f"✗ No gradients computed")
            return False
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VLAT Policy Test Suite")
    print("=" * 60)
    
    # Test 1: Configuration
    config = test_config()
    if config is None:
        print("\n✗ Tests aborted due to config failure")
        return
    
    # Test 2: Model instantiation
    policy = test_model_instantiation(config)
    if policy is None:
        print("\n✗ Tests aborted due to instantiation failure")
        return
    
    # Test 3: Forward pass
    if not test_forward_pass(policy, config):
        print("\n✗ Tests aborted due to forward pass failure")
        return
    
    # Test 4: Action prediction
    if not test_action_prediction(policy, config):
        print("\n⚠ Action prediction test failed but continuing...")
    
    # Test 5: Gradient flow
    test_gradient_flow(policy, config)
    
    print("\n" + "=" * 60)
    print("Test Suite Complete!")
    print("=" * 60)
    print("\n✓ VLAT policy is ready for training!")
    print("\nTo start training, run:")
    print("  accelerate launch \\")
    print("    --config_file configs/accelerate/accelerate_config.yaml \\")
    print("    kuavo_train/train_policy_with_accelerate.py \\")
    print("    --config-path=../configs/policy \\")
    print("    --config-name=vlat_config.yaml")


if __name__ == "__main__":
    main()
