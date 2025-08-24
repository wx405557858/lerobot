#!/usr/bin/env python3
"""
Extended test script that mimics the actual training setup more closely,
including delta timestamps and other training-specific configurations.
"""

import sys
import torch
from pathlib import Path

# Add the lerobot package to the path
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader


def test_with_delta_timestamps():
    """Test with delta timestamps that might be used in training."""
    print("Testing dataset loading with delta timestamps...")
    
    # Parameters from your training command
    repo_id = "lerobot/svla_so100_stacking"
    batch_size = 64
    
    # Common delta timestamps used in training (example values)
    delta_timestamps = {
        "observation.images.top": [-0.1, -0.03333, 0.0],  # 3 frames back
        "observation.images.wrist": [-0.1, -0.03333, 0.0],  # 3 frames back
        "action": [0.0],  # current frame only
        "observation.state": [0.0],  # current frame only
    }
    
    try:
        # Load the dataset with delta timestamps
        print(f"Loading dataset: {repo_id}")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
            download_videos=True,
        )
        
        print(f"Dataset loaded successfully with delta timestamps!")
        print(f"Number of episodes: {dataset.num_episodes}")
        print(f"Number of frames: {dataset.num_frames}")
        print(f"Delta indices: {dataset.delta_indices}")
        
        # Test creating a DataLoader
        print(f"\nCreating DataLoader with batch_size={batch_size}...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Use fewer workers for testing
            pin_memory=True,
        )
        
        print("DataLoader created successfully!")
        
        # Test loading a batch
        print("\nTesting batch loading with delta timestamps...")
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("\n✅ Delta timestamps test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during delta timestamps test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_loading():
    """Test basic dataset loading without delta timestamps."""
    print("Testing basic dataset loading...")
    
    repo_id = "lerobot/svla_so100_stacking"
    batch_size = 64
    
    try:
        dataset = LeRobotDataset(repo_id=repo_id)
        print(f"✅ Basic loading test passed!")
        
        # Quick batch test
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        batch = next(iter(dataloader))
        print(f"✅ Basic batch loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic loading test failed: {e}")
        return False


def main():
    print("Running comprehensive dataset loading tests...\n")
    
    # Test 1: Basic loading
    basic_success = test_basic_loading()
    print()
    
    # Test 2: Delta timestamps
    delta_success = test_with_delta_timestamps()
    print()
    
    if basic_success and delta_success:
        print("🎉 All tests passed! The dataset loading fix is working correctly.")
        print("You should now be able to run your training command successfully.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
