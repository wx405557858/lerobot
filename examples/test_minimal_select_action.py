#!/usr/bin/env python3
"""
Minimal test script for SmolVLA select_action functionality.
"""

import sys
import torch
from pathlib import Path

# Add the lerobot package to the path
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from torch.utils.data import DataLoader


def test_select_action():
    """Minimal test for SmolVLA select_action functionality."""
    print("Testing SmolVLA select_action...\n")
    
    # Configuration
    policy_path = "lerobot/smolvla_base"
    dataset_repo_id = "lerobot/svla_so100_stacking"
    
    try:
        # 1. Create config
        dataset_config = DatasetConfig(repo_id=dataset_repo_id)
        policy_config = SmolVLAConfig(device="cuda")
        policy_config.pretrained_path = policy_path
        
        train_config = TrainPipelineConfig(
            dataset=dataset_config,
            policy=policy_config,
            batch_size=2,
            steps=100,
            output_dir=Path("./test_outputs"),
        )
        
        # 2. Load dataset and policy
        print("Loading dataset and policy...")
        dataset = make_dataset(train_config)
        policy = make_policy(cfg=train_config.policy, ds_meta=dataset.meta)
        print(f"✅ Policy loaded: {type(policy).__name__}")
        
        # 3. Prepare observation batch
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        
        # Move to device and prepare observation-only batch
        device = next(policy.parameters()).device
        obs_batch = {}
        for key, value in batch.items():
            if not key.startswith('action') and 'action' not in key:
                if isinstance(value, torch.Tensor):
                    obs_batch[key] = value.to(device)
                else:
                    obs_batch[key] = value
        
        print(f"Observation keys: {list(obs_batch.keys())}")
        
        # 4. Test select_action
        policy.eval()
        with torch.no_grad():
            actions = policy.select_action(obs_batch)
            print(f"✅ Action selection successful!")
            print(f"Actions: {actions}")
            print(f"Action shape: {actions.shape}")
            print(f"Action values: {actions[0].cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_select_action()
    if success:
        print(f"\n🎉 select_action test passed!")
    else:
        print(f"\n❌ select_action test failed!")
    sys.exit(0 if success else 1)
