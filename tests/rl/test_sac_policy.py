#!/usr/bin/env python

"""
Minimal SAC Policy Test Script

This script demonstrates how to:
1. Load the wx405557858/pick_ring_env_7 dataset from HuggingFace Hub
2. Initialize a SAC (Soft Actor-Critic) policy
3. Use the policy to select actions from observations

Usage:
    conda activate lerobot
    python test_sac_policy.py
"""

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig

def main():
    # Dataset to load
    repo_id = "wx405557858/pick_ring_env_7"
    
    print(f"Loading dataset: {repo_id}")
    
    # Load dataset metadata to understand structure
    ds_meta = LeRobotDatasetMetadata(repo_id)
    print(f"Dataset has {ds_meta.total_episodes} episodes with {ds_meta.total_frames} total frames")
    
    # Load a small subset of the dataset for testing
    dataset = LeRobotDataset(repo_id, episodes=[0, 1, 2])
    print(f"Loaded {dataset.num_episodes} episodes for testing")
    
    # Examine dataset structure
    sample = dataset[0]
    print(f"\nDataset sample contains:")
    for key, value in sample.items():
        if key.startswith('observation') and isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Create input/output feature mappings for SAC config
    input_features = {}
    output_features = {}
    
    for key, feature in ds_meta.features.items():
        if key.startswith('observation'):
            # Create dummy tensors with correct shapes for config
            input_features[key] = torch.zeros(feature['shape'])
        elif key == 'action':
            output_features[key] = torch.zeros(feature['shape'])
    
    print(f"\nInput features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")
    
    # Create SAC configuration
    config = SACConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        shared_encoder=True,
        num_critics=2,
        latent_dim=256,
        use_text_prompt=True,  # Enable text prompt support
        text_encoder_name="openai/clip-vit-base-patch32",
        freeze_text_encoder=True,
        text_embedding_dim=512,
    )
    
    # Initialize SAC policy
    policy = SACPolicy(config=config)
    print(f"\n✓ SAC policy initialized successfully with text prompt support")
    
    # Test action selection
    print(f"\nTesting action selection with text prompts...")
    
    # Prepare batch from dataset sample
    batch = {}
    for key, value in sample.items():
        if key.startswith('observation') and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)  # Add batch dimension
    
    # Add task information for text prompt
    if 'task' in sample:
        batch['task'] = sample['task']  # Task is already a string
        task_description = sample['task']
    else:
        batch['task'] = "pick up the ring"  # Default task description
        task_description = "pick up the ring"
    
    print(f"Using task description: '{task_description}'")
    
    # Select actions using the policy
    print(f"\nSelecting actions from observations:")
    for i in range(5):
        with torch.no_grad():
            action = policy.select_action(batch)
        
        # The action contains continuous values for: [delta_x_ee, delta_y_ee, delta_z_ee, gripper_delta]
        action_values = action.squeeze().numpy()
        print(f"  Action {i+1}: [{action_values[0]:6.3f}, {action_values[1]:6.3f}, {action_values[2]:6.3f}, {action_values[3]:6.3f}]")
    
    print(f"\n✓ Action selection completed successfully!")
    print(f"\nThe SAC policy is working and can generate actions from the dataset observations.")

if __name__ == "__main__":
    main()
