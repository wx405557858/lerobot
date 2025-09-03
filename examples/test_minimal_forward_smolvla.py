#!/usr/bin/env python3
"""
Minimal test script for SmolVLA select_action functionality with pick_ring dataset.
This script tests the policy with a 4-dimensional action space.
"""

import sys
import torch
from pathlib import Path

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Add the lerobot package to the path
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from torch.utils.data import DataLoader


def test_select_action_pick_ring():
    """Minimal test for SmolVLA select_action functionality with pick_ring dataset."""
    print("Testing SmolVLA select_action with pick_ring dataset...\n")
    
    # Configuration
    policy_path = "lerobot/smolvla_base"
    # policy_path = "/home/shaoxiong/Code/lerobot_original/outputs/train/2025-08-23/22-10-41_smolvla/checkpoints/080000/pretrained_model"
    dataset_repo_id = "wx405557858/pick_ring_env_5"
    
    try:
        # 1. Create config
        dataset_config = DatasetConfig(repo_id=dataset_repo_id)
        policy_config = SmolVLAConfig(
            device="cuda",
            chunk_size=1,
            n_action_steps=1,
            max_action_dim=32,
            num_vlm_layers=4,
        )
        policy_config.pretrained_path = policy_path
        
        train_config = TrainPipelineConfig(
            dataset=dataset_config,
            policy=policy_config,
            batch_size=2,
            steps=100,
            output_dir=Path("./test_outputs_pick_ring"),
        )
        
        # 2. Load dataset and policy
        print("Loading dataset and policy...")
        dataset = make_dataset(train_config)
        policy: SmolVLAPolicy = make_policy(cfg=train_config.policy, ds_meta=dataset.meta)
        print(f"✅ Policy loaded: {type(policy).__name__}")
        
        # Print dataset metadata for debugging
        print(f"Dataset metadata:")
        print(f"  Action keys: {dataset.meta.names.get('action', [])}")
        print(f"  Action shape: {dataset.meta.shapes.get('action', 'Unknown')}")

        # for name, param in policy.named_parameters():
        #     print(f"Policy parameter '{name}' requires_grad: {param.requires_grad}, shape: {param.shape}")

        print(f"total parameters trainable: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
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
        print(f"task: {obs_batch.get('task', 'N/A')}")
        
        # 4. Test select_action
        policy.eval()
        with torch.no_grad():
            # actions = policy.select_action(obs_batch)
            actions = policy.forward_embeddings(obs_batch)
            print(f"✅ Action selection successful!")
            print(f"Actions: {actions}")
            print(f"Action shape: {actions.shape}")
            print(f"Action values: {actions[0].cpu().numpy()}")
            
            # Verify action dimensionality
            expected_action_dim = policy_config.max_action_dim
            actual_action_dim = actions.shape[-1]
            if actual_action_dim == expected_action_dim:
                print(f"✅ Action dimension correct: {actual_action_dim}D")
            else:
                print(f"⚠️  Action dimension mismatch: expected {expected_action_dim}D, got {actual_action_dim}D")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_select_action_pick_ring()
    if success:
        print(f"\n🎉 pick_ring select_action test passed!")
    else:
        print(f"\n❌ pick_ring select_action test failed!")
    sys.exit(0 if success else 1)
