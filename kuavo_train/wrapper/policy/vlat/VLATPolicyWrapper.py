"""
VLAT Policy Wrapper.

Wraps the VLAT model to conform to the LeRobot policy interface.
Handles:
- Image preprocessing and resizing
- Multi-modal batch preparation
- Loss computation with multiple objectives
- Action prediction and chunking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict
from pathlib import Path
from collections import deque
import os
import builtins
from typing import TypeVar

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from kuavo_train.wrapper.policy.vlat.VLATConfigWrapper import CustomVLATConfigWrapper
from kuavo_train.wrapper.policy.vlat.VLATModelWrapper import VLATModel

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError

T = TypeVar("T", bound="CustomVLATPolicyWrapper")
OBS_DEPTH = "observation.depth"


class CustomVLATPolicyWrapper(PreTrainedPolicy):
    """
    Vision-Language-Action Transformer (VLAT) Policy.
    
    A multi-modal transformer policy with:
    - ViT visual encoding
    - Perceiver multi-modal fusion
    - Temporal sequence modeling
    - Action chunking with uncertainty estimation
    """
    
    name = "custom_vlat"
    config_class = CustomVLATConfigWrapper
    
    def __init__(self, config: CustomVLATConfigWrapper):
        super().__init__(config)
        
        self.config = config
        self.model = VLATModel(config)
        
        # Action chunking state
        self.reset()
    
    def reset(self):
        """Reset policy state (called on environment reset)."""
        self._action_queue = deque(maxlen=self.config.chunk_size)
        self._step_count = 0
    
    def _prepare_observation_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Prepare observations for model input.
        
        Handles:
        - Grouping image features
        - Grouping depth features
        - Image resizing if configured
        """
        batch = dict(batch)  # Shallow copy
        
        # Group image features
        if self.config.image_features:
            images = []
            for key in self.config.image_features:
                img = batch[key]
                
                # Resize if configured
                if self.config.image_size is not None:
                    target_h, target_w = self.config.image_size
                    if img.shape[-2:] != (target_h, target_w):
                        # Assuming img is (B, n_obs_steps, C, H, W) or similar
                        original_shape = img.shape
                        if img.dim() == 5:
                            # (B, n_obs_steps, C, H, W)
                            B, T, C, H, W = img.shape
                            img = img.reshape(B * T, C, H, W)
                            img = F.interpolate(
                                img,
                                size=(target_h, target_w),
                                mode='bilinear',
                                align_corners=False
                            )
                            img = img.reshape(B, T, C, target_h, target_w)
                        elif img.dim() == 4:
                            # (B, C, H, W)
                            img = F.interpolate(
                                img,
                                size=(target_h, target_w),
                                mode='bilinear',
                                align_corners=False
                            )
                
                images.append(img)
            
            # Stack images: (B, n_obs_steps, n_cameras, C, H, W)
            if images[0].dim() == 4:
                # Single timestep: (B, C, H, W) -> (B, 1, C, H, W)
                images = [img.unsqueeze(1) for img in images]
            
            batch[OBS_IMAGES] = torch.stack(images, dim=2)
        
        # Group depth features
        if getattr(self.config, 'use_depth', False) and self.config.depth_features:
            depths = []
            for key in self.config.depth_features:
                depth = batch[key]
                
                # Convert to single channel if needed (average RGB channels)
                if depth.dim() >= 3 and depth.shape[-3] == 3:
                    depth = depth.mean(dim=-3, keepdim=True)
                
                # Resize if configured
                if self.config.image_size is not None:
                    target_h, target_w = self.config.image_size
                    if depth.shape[-2:] != (target_h, target_w):
                        original_shape = depth.shape
                        if depth.dim() == 5:
                            # (B, n_obs_steps, C, H, W)
                            B, T, C, H, W = depth.shape
                            depth = depth.reshape(B * T, C, H, W)
                            depth = F.interpolate(
                                depth,
                                size=(target_h, target_w),
                                mode='bilinear',
                                align_corners=False
                            )
                            depth = depth.reshape(B, T, C, target_h, target_w)
                        elif depth.dim() == 4:
                            # (B, C, H, W)
                            depth = F.interpolate(
                                depth,
                                size=(target_h, target_w),
                                mode='bilinear',
                                align_corners=False
                            )
                
                depths.append(depth)
            
            # Stack depths
            if depths[0].dim() == 4:
                depths = [d.unsqueeze(1) for d in depths]
            
            batch[OBS_DEPTH] = torch.stack(depths, dim=2)
        
        return batch
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Forward pass for training.
        
        Computes multiple loss components:
        - Behavior cloning (MSE) loss
        - Action smoothness loss
        - Uncertainty regularization loss
        - Optional contrastive loss
        
        Args:
            batch: Dictionary with observations and actions
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Prepare batch
        batch = self._prepare_observation_batch(batch)
        
        # Forward through model
        actions_pred, uncertainty = self.model(batch)  # (B, chunk_size, action_dim)
        actions_gt = batch[ACTION]  # (B, chunk_size, action_dim)
        
        # 1. Behavior cloning loss (MSE)
        action_mask = ~batch.get("action_is_pad", torch.zeros_like(actions_gt[:, :, 0]).bool())
        
        mse_loss = F.mse_loss(
            actions_pred * action_mask.unsqueeze(-1),
            actions_gt * action_mask.unsqueeze(-1),
            reduction='sum'
        ) / (action_mask.sum() * actions_gt.shape[-1] + 1e-8)
        
        loss_dict = {
            "mse_loss": mse_loss.item(),
        }
        
        total_loss = mse_loss
        
        # 2. Action smoothness loss (penalize large action changes)
        if self.config.lambda_smooth > 0:
            action_diff = actions_pred[:, 1:] - actions_pred[:, :-1]
            smooth_loss = (action_diff ** 2).mean()
            total_loss = total_loss + self.config.lambda_smooth * smooth_loss
            loss_dict["smooth_loss"] = smooth_loss.item()
        
        # 3. Uncertainty regularization (prevent overconfident predictions)
        if self.config.use_uncertainty and uncertainty is not None:
            # Encourage reasonable uncertainty levels
            uncertainty_reg = torch.mean(
                (uncertainty - 0.1) ** 2  # Target uncertainty around 0.1
            )
            total_loss = total_loss + self.config.lambda_uncertainty * uncertainty_reg
            loss_dict["uncertainty_reg"] = uncertainty_reg.item()
            loss_dict["mean_uncertainty"] = uncertainty.mean().item()
            
            # Also compute uncertainty-weighted MSE
            uncertainty_weight = 1.0 / (uncertainty ** 2 + 1e-6)
            weighted_mse = (
                (actions_pred - actions_gt) ** 2 * uncertainty_weight * action_mask.unsqueeze(-1)
            ).sum() / (action_mask.sum() * actions_gt.shape[-1] + 1e-8)
            
            # Add log(uncertainty) regularization term (from uncertainty loss formulation)
            log_uncertainty = torch.log(uncertainty + 1e-6)
            log_unc_loss = (log_uncertainty * action_mask.unsqueeze(-1)).sum() / (action_mask.sum() * actions_gt.shape[-1] + 1e-8)
            
            uncertainty_loss = weighted_mse + log_unc_loss
            total_loss = total_loss + 0.1 * uncertainty_loss
            loss_dict["uncertainty_loss"] = uncertainty_loss.item()
        
        # 4. Contrastive loss (optional - for future implementation)
        if self.config.lambda_contrast > 0:
            # Placeholder for contrastive learning
            # This would require temporal augmentation and positive/negative pairs
            pass
        
        return total_loss, loss_dict
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select a single action for execution in the environment.
        
        Uses action chunking: predicts a chunk of actions and executes them
        one by one until the queue is empty, then predicts a new chunk.
        
        Args:
            batch: Dictionary with current observations
        
        Returns:
            (B, action_dim) single action to execute
        """
        self.eval()
        
        # Check if we need to predict a new action chunk
        if len(self._action_queue) == 0 or self._step_count % self.config.n_action_steps == 0:
            # Predict new action chunk
            action_chunk = self.predict_action_chunk(batch)  # (B, chunk_size, action_dim)
            
            # Add to queue
            self._action_queue.clear()
            for t in range(self.config.chunk_size):
                self._action_queue.append(action_chunk[:, t])
        
        # Pop next action from queue
        action = self._action_queue.popleft()
        self._step_count += 1
        
        return action
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Predict a full chunk of actions.
        
        Args:
            batch: Dictionary with observations
        
        Returns:
            (B, chunk_size, action_dim) predicted actions
        """
        self.eval()
        
        # Prepare batch
        batch = self._prepare_observation_batch(batch)
        
        # Forward through model
        actions, uncertainty = self.model(batch)
        
        return actions
    
    def get_optim_params(self) -> list[dict]:
        """
        Return optimizer parameter groups.
        
        Separate learning rates for:
        - Vision encoder (ViT) parameters
        - Other parameters
        """
        vit_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Lower learning rate for pretrained ViT encoders
            if 'rgb_encoder' in name or 'depth_encoder' in name:
                vit_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {
                "params": other_params,
                "lr": self.config.optimizer_lr,
            },
            {
                "params": vit_params,
                "lr": self.config.optimizer_lr * 0.1,  # 10x lower for pretrained encoders
            },
        ]
    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: CustomVLATConfigWrapper | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        Load pretrained policy from checkpoint.
        
        The policy is set in evaluation mode by default.
        To train it, call `policy.train()`.
        """
        if config is None:
            config = CustomVLATConfigWrapper.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e
        
        policy.to(config.device)
        policy.eval()
        return policy
