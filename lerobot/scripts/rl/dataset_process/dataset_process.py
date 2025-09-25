#!/usr/bin/env python

"""
Dataset Processing Script

This script demonstrates how to:
1. Load a dataset from HuggingFace Hub
2. Trim episodes (select a subset)
3. Create a new LeRobot dataset with the trimmed episodes
4. Save the new dataset locally (and optionally to the Hub)

Usage:
    conda activate lerobot
    python dataset_process.py
"""

import torch
import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig
import draccus
import shutil
from pathlib import Path

from lerobot.scripts.rl.learner import check_nan_in_transition, get_observation_features

@dataclasses.dataclass
class ProcessDatasetConfig:
    repo_id: str = "wx405557858/pick_ring_env_38_tmp"
    device: str = "cuda"
    num_training_steps: int = 10000
    log_interval: int = 10
    batch_size: int = 128

def trim_dataset(config: ProcessDatasetConfig):
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

    # Load a small subset of the dataset for testing
    dataset = LeRobotDataset(repo_id, episodes=list(range(1800, ds_meta.total_episodes)))
    print(f"Loaded {dataset.num_episodes} episodes for testing")
    
    # Examine dataset structure
    sample = dataset[0]
    print(f"\nDataset sample contains:")
    print(f"  action: {sample['action'].shape if hasattr(sample['action'], 'shape') else sample['action']}")
    print(f"  task: {sample['task']}")
    print(f"  episode_index: {sample['episode_index']}")
    
    # Show available observation keys
    for sample in dataset:
        for key, value in sample.items():
            if key.startswith('observation') and isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
                if key.startswith('observation.images'):
                    import cv2
                    cv2.imshow(key, value.permute(1, 2, 0).numpy()[:, :, ::-1])  # Show BGR channels
                    cv2.waitKey(0)
    
    print(f"\nStarting to create trimmed dataset...")
    
    # make new dataset with trimmed episodes
    new_repo_id = repo_id + "_trimmed_test"
    print(f"Saving trimmed dataset to: {new_repo_id}")

    # Clean up any existing dataset directory
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    dataset_path = Path(HF_LEROBOT_HOME) / new_repo_id

    # # Create a new (empty) LeRobotDataset for writing
    # new_dataset = LeRobotDataset.create(
    #     repo_id=new_repo_id,
    #     fps=dataset.fps,
    #     root=None,  # Will use default cache directory
    #     robot_type=dataset.meta.robot_type,
    #     features=dataset.meta.info["features"],
    #     use_videos=len(dataset.meta.video_keys) > 0,
    # )
    
    # print(f"Created new dataset with {new_dataset.meta.robot_type} robot type")
    # print(f"Dataset features: {list(new_dataset.features.keys())}")
    
    # # Iterate through the trimmed dataset and copy frames to new dataset
    # prev_episode_index = -1
    # task_name = "Pick ring environment task"  # Default task name
    
    # for frame_idx in range(len(dataset)):
    #     frame = dataset[frame_idx]
        
    #     # Create a copy of the frame to add to the new dataset
    #     new_frame = {}
    #     for key, value in frame.items():
    #         # Skip metadata fields that will be automatically handled
    #         if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
    #             continue
            
    #         # Handle reward and done signals - ensure they have proper shape
    #         if key in ("next.done", "next.reward"):
    #             if hasattr(value, 'shape') and len(value.shape) == 0:
    #                 value = value.unsqueeze(0)
            
    #         # Handle complementary info fields that might be scalars
    #         if key.startswith("complementary_info.") and hasattr(value, 'shape') and len(value.shape) == 0:
    #             value = value.unsqueeze(0)
            
    #         new_frame[key] = value
        
    #     # Get task from original frame if available
    #     if "task" in frame:
    #         current_task = frame["task"]
    #     else:
    #         current_task = task_name
            
    #     # Add frame to new dataset
    #     new_dataset.add_frame(new_frame, task=current_task)
        
    #     # Check if we've moved to a new episode
    #     current_episode_index = frame["episode_index"].item()
    #     if current_episode_index != prev_episode_index and prev_episode_index != -1:
    #         # Save the previous episode
    #         new_dataset.save_episode()
    #         print(f"Saved episode {prev_episode_index}")
        
    #     prev_episode_index = current_episode_index
    
    # # Save the last episode
    # new_dataset.save_episode()
    # print(f"Saved final episode {prev_episode_index}")
    
    # print(f"New dataset created with {new_dataset.num_episodes} episodes and {new_dataset.num_frames} frames")
    
    # # Optionally push to hub (set to False by default for safety)
    # push_to_hub = False
    # if push_to_hub:
    #     print("Pushing trimmed dataset to Hugging Face Hub...")
    #     new_dataset.push_to_hub()
    #     print(f"Dataset uploaded to: https://huggingface.co/datasets/{new_repo_id}")
    # else:
    #     print(f"Dataset saved locally. To upload to hub, set push_to_hub=True")

@draccus.wrap()
def main(config: ProcessDatasetConfig):
    trim_dataset(config)

if __name__ == "__main__":
    main()
