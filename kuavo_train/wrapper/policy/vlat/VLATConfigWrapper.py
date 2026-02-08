from typing import Any, Dict
from dataclasses import dataclass, fields, field
import copy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from omegaconf import DictConfig, OmegaConf, ListConfig
from copy import deepcopy
from pathlib import Path
import draccus
from huggingface_hub.constants import CONFIG_NAME
from typing import TypeVar

T = TypeVar("T", bound="CustomVLATConfigWrapper")

@PreTrainedConfig.register_subclass("custom_vlat")
@dataclass
class CustomVLATConfigWrapper(PreTrainedConfig):
    """
    Vision-Language-Action Transformer (VLAT) Configuration
    
    This config defines a multi-modal transformer policy that uses:
    - Vision Transformer (ViT) for visual encoding
    - Perceiver-style fusion for multi-modal integration
    - Temporal Transformer for sequence modeling
    - Transformer Decoder for action prediction with uncertainty estimation
    """
    
    # Observation configuration
    n_obs_steps: int = 2
    chunk_size: int = 100
    n_action_steps: int = 1
    
    # Vision Transformer configuration
    vision_backbone: str = "vit_small_patch16_224"
    pretrained_backbone_weights: str | None = None
    patch_size: int = 16
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    vit_embed_dim: int = 384  # ViT-Small default
    
    # Perceiver fusion configuration
    n_query_tokens: int = 64
    fusion_dim: int = 512
    fusion_heads: int = 8
    fusion_layers: int = 2
    fusion_dropout: float = 0.1
    
    # Temporal modeling configuration
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_dim: int = 512
    temporal_dropout: float = 0.1
    use_causal_temporal_attention: bool = False
    
    # Action prediction configuration
    decoder_layers: int = 4
    decoder_heads: int = 8
    decoder_dim: int = 512
    decoder_dropout: float = 0.1
    
    # Training configuration
    use_uncertainty: bool = True
    lambda_smooth: float = 0.1
    lambda_contrast: float = 0.01
    lambda_uncertainty: float = 0.001
    
    # Optimizer configuration
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    
    # Scheduler configuration
    scheduler_name: str = "cosine"
    
    # Normalization mapping
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)
    
    # Mixed precision
    use_amp: bool = False
    
    # Custom settings (for depth, etc.)
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        
        # Set default normalization
        default_map = {
            "RGB": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.MEAN_STD,
            "DEPTH": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
        
        # Merge and update the normalization_mapping
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged
        
        # Process custom settings
        if isinstance(self.custom, (DictConfig, dict)):
            for k, v in self.custom.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(
                        f"Custom setting '{k}: {v}' conflicts with the parent base configuration. "
                        f"Remove it from 'custom' and modify in the parent configuration instead."
                    )
        
        self._convert_omegaconf_fields()
    
    def _convert_omegaconf_fields(self):
        """Convert OmegaConf objects to standard Python types"""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Return all RGB/visual features"""
        # Support both RGB and VISUAL types (RGB from custom patches, VISUAL from standard lerobot)
        try:
            rgb_type = FeatureType.RGB
        except AttributeError:
            rgb_type = None
        
        if rgb_type is not None:
            return {
                key: ft 
                for key, ft in self.input_features.items() 
                if ft.type in (rgb_type, FeatureType.VISUAL)
            }
        else:
            return {
                key: ft 
                for key, ft in self.input_features.items() 
                if ft.type == FeatureType.VISUAL
            }
    
    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        """Return all depth features"""
        # DEPTH type might not exist in standard lerobot
        try:
            depth_type = FeatureType.DEPTH
        except AttributeError:
            return {}
        
        return {
            key: ft 
            for key, ft in self.input_features.items() 
            if ft.type == depth_type
        }
    
    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Indices of observations relative to current timestep"""
        return list(range(-self.n_obs_steps + 1, 1))
    
    @property
    def action_delta_indices(self) -> list[int] | None:
        """Indices of actions relative to current timestep"""
        return list(range(self.chunk_size))
    
    @property
    def reward_delta_indices(self) -> list[int] | None:
        """VLAT doesn't use reward"""
        return None

    def validate_features(self) -> None:
        """Validate that input features are properly configured"""
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        # Check that all input images have the same shape
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        f"but we expect all image shapes to match."
                    )
        
        # Check depth features
        if len(self.depth_features) == 0:
            print("No depth features found!")
        else:
            first_depth_key, first_depth_ft = next(iter(self.depth_features.items()))
            for key, depth_ft in self.depth_features.items():
                if depth_ft.shape != first_depth_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_depth_key}`, "
                        f"but we expect all depth shapes to match."
                    )

    def get_optimizer_preset(self):
        """Return optimizer configuration"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
    
    def get_scheduler_preset(self):
        """Return learning rate scheduler configuration"""
        if self.scheduler_name == "cosine":
            return DiffuserSchedulerConfig(
                name="cosine",
                num_warmup_steps=500,
            )
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save configuration to directory"""
        cfg_copy = deepcopy(self)
        
        # Remove custom attributes that were added dynamically
        if isinstance(cfg_copy.custom, dict):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        elif hasattr(cfg_copy, "custom") and hasattr(cfg_copy.custom, "keys"):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)
    
    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """Load configuration from pretrained model"""
        parent_cls = PreTrainedConfig
        return parent_cls.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )
