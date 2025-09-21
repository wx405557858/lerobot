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
import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig
import cv2
import draccus

from lerobot.scripts.rl.learner import check_nan_in_transition, get_observation_features

@dataclasses.dataclass
class PretrainingConfig:
    repo_id: str = "wx405557858/pick_ring_env_36"
    device: str = "cuda"
    num_training_steps: int = 10000
    log_interval: int = 10
    batch_size: int = 128

def load_dataset(config: PretrainingConfig):
    # Dataset to load
    repo_id = config.repo_id

    print(f"Loading dataset: {repo_id}")
    
    # Load dataset metadata to understand structure
    ds_meta = LeRobotDatasetMetadata(repo_id)
    print(f"Dataset has {ds_meta.total_episodes} episodes with {ds_meta.total_frames} total frames")
    
    
    # Then we grab all the image frames from the first camera:
    camera_key = ds_meta.camera_keys

    # # The objects returned by the dataset are all torch.Tensors
    # print(type(frames[0]))
    # print(frames[0].shape)

    print(f"camera_key: {camera_key}")
    delta_timestamps = {}

    delta_timestamps["action"] = [-0.5, 0.0]  # 500 ms before and current action
    delta_timestamps["observation.state"] = [-0.5, 0.0]  # 500 ms before and current state
    for key in camera_key:
        delta_timestamps[key] = [-0.5, 0.0]  # 500 ms before and current image

    # Load a small subset of the dataset for testing
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    print(f"Loaded {dataset.num_episodes} episodes for testing")
    
    # Examine dataset structure
    sample = dataset[0]
    print(f"\nDataset sample contains:")
    print(f"  action: {sample['action']}")
    print(f"  task: {sample['task']}")
    print(f"sample: {sample}")
    for key, value in sample.items():
        if key.startswith('observation') and isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
            # if "image" in key:
            #     cv2.imshow(key, cv2.cvtColor(value.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))  # Show RGB channels
            #     cv2.waitKey(200)  # Display each image for 500 ms
    
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
        device=config.device,
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
    policy.to(config.device)
    print(f"\n✓ SAC policy initialized successfully with text prompt support")
    return dataset, policy

def train(config: PretrainingConfig, dataset: LeRobotDataset, policy: SACPolicy):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    policy.train()

    camera_keys = dataset.meta.camera_keys
    iter_step = 0
    mean_loss = 0.0
    while True:
        for batch in dataloader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(config.device)

            actions = batch["action"][:, 0]
            rewards = batch["next.reward"]
            observations = {"observation.state": batch["observation.state"][:, 0]}
            for key in camera_keys:
                observations[key] = batch[key][:, 0]
            observations_with_task = observations.copy()
            observations_with_task["task"] = batch["task"]
            next_observations = {"observation.state": batch["observation.state"][:, 1]}
            for key in camera_keys:
                next_observations[key] = batch[key][:, 1]
            next_observations_with_task = next_observations.copy()
            next_observations_with_task["task"] = batch["task"]
            done = batch["next.done"]
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations_with_task, next_observations=next_observations_with_task
            )

            predicted_action, inverse_dynamics_loss = policy.forward_inverse_dynamics(
                observations_with_task,
                actions,
                next_observations_with_task,
                observation_features,
                next_observation_features,
            )

            optimizer.zero_grad()
            inverse_dynamics_loss.backward()
            optimizer.step()

            mean_loss = 0.9 * mean_loss + 0.1 * inverse_dynamics_loss.detach().cpu().item()

            iter_step += 1
            if iter_step >= config.num_training_steps:
                break
            if iter_step % config.log_interval == 0:
                print(f"Step {iter_step}: mean inverse dynamics loss = {mean_loss:.6f}")



@draccus.wrap()
def main(config: PretrainingConfig):
    dataset, policy = load_dataset(config)
    train(config, dataset, policy)

if __name__ == "__main__":
    main()
