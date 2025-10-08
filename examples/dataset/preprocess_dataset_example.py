#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating how to use the dataset preprocessing utilities.

This script shows how to:
1. Load a LeRobot dataset
2. Downsample the FPS
3. Resize images
4. Delete specific episodes
5. Chain multiple preprocessing operations

Usage:
    python examples/dataset/preprocess_dataset_example.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.preprocess_dataset import delete_episodes, downsample_dataset, resize_dataset


def example_downsample():
    """Example: Downsample a dataset from 30 fps to 15 fps."""
    print("\n=== Example 1: Downsample FPS ===")

    # Load original dataset
    original = LeRobotDataset("lerobot/pusht")
    print(f"Original dataset: {original.fps} fps, {original.num_frames} frames")

    # Downsample to 15 fps
    downsampled = downsample_dataset(
        original_dataset=original,
        target_fps=15,
        new_repo_id="lerobot/pusht_15fps",
        new_dataset_root="/tmp/pusht_15fps",
        push_to_hub=False,
    )
    print(f"Downsampled dataset: {downsampled.fps} fps, {downsampled.num_frames} frames")
    print(f"Saved to: {downsampled.root}")


def example_resize():
    """Example: Resize images in a dataset to 128x128."""
    print("\n=== Example 2: Resize Images ===")

    # Load original dataset
    original = LeRobotDataset("lerobot/pusht")
    camera_key = original.meta.camera_keys[0]
    original_shape = original.meta.features[camera_key]["shape"]
    print(f"Original image shape: {original_shape}")

    # Resize to 128x128
    resized = resize_dataset(
        original_dataset=original,
        resize_size=(128, 128),
        new_repo_id="lerobot/pusht_128x128",
        new_dataset_root="/tmp/pusht_128x128",
        push_to_hub=False,
    )
    new_shape = resized.meta.features[camera_key]["shape"]
    print(f"Resized image shape: {new_shape}")
    print(f"Saved to: {resized.root}")


def example_delete_episodes():
    """Example: Delete specific episodes from a dataset."""
    print("\n=== Example 3: Delete Episodes ===")

    # Load original dataset
    original = LeRobotDataset("lerobot/pusht")
    print(f"Original dataset: {original.num_episodes} episodes, {original.num_frames} frames")

    # Delete episodes 0, 5, and 10
    filtered = delete_episodes(
        original_dataset=original,
        episodes_to_delete=[0, 5, 10],
        new_repo_id="lerobot/pusht_filtered",
        new_dataset_root="/tmp/pusht_filtered",
        push_to_hub=False,
    )
    print(f"Filtered dataset: {filtered.num_episodes} episodes, {filtered.num_frames} frames")
    print(f"Saved to: {filtered.root}")


def example_combined():
    """Example: Chain multiple preprocessing operations."""
    print("\n=== Example 4: Combined Operations ===")

    # Load original dataset
    original = LeRobotDataset("lerobot/pusht")
    print(f"Original: {original.fps} fps, {original.num_episodes} episodes, {original.num_frames} frames")

    # Step 1: Delete some episodes
    print("\nStep 1: Deleting episodes...")
    step1 = delete_episodes(
        original_dataset=original,
        episodes_to_delete=[0, 1, 2],
        new_repo_id="temp/step1",
        new_dataset_root="/tmp/preprocess_chain/step1",
    )
    print(f"After deletion: {step1.num_episodes} episodes, {step1.num_frames} frames")

    # Step 2: Downsample FPS
    print("\nStep 2: Downsampling FPS...")
    step2 = downsample_dataset(
        original_dataset=step1,
        target_fps=15,
        new_repo_id="temp/step2",
        new_dataset_root="/tmp/preprocess_chain/step2",
    )
    print(f"After downsampling: {step2.fps} fps, {step2.num_frames} frames")

    # Step 3: Resize images
    print("\nStep 3: Resizing images...")
    final = resize_dataset(
        original_dataset=step2,
        resize_size=(64, 64),
        new_repo_id="lerobot/pusht_preprocessed",
        new_dataset_root="/tmp/preprocess_chain/final",
    )
    camera_key = final.meta.camera_keys[0]
    final_shape = final.meta.features[camera_key]["shape"]
    print(f"Final dataset: {final.fps} fps, {final.num_episodes} episodes, {final.num_frames} frames")
    print(f"Image shape: {final_shape}")
    print(f"Saved to: {final.root}")


if __name__ == "__main__":
    print("LeRobot Dataset Preprocessing Examples")
    print("=" * 50)

    # Run individual examples (uncomment the ones you want to try)
    # Note: These examples require the pusht dataset to be downloaded

    # example_downsample()
    # example_resize()
    # example_delete_episodes()
    # example_combined()

    print(
        "\nNote: To run these examples, uncomment the function calls above "
        "and ensure the pusht dataset is available."
    )
