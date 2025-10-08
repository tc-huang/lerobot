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
Utilities for preprocessing LeRobot datasets:
- Downsample FPS
- Resize images
- Delete specific episodes
"""

import logging
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD


def downsample_dataset(
    original_dataset: LeRobotDataset,
    target_fps: int,
    new_repo_id: str,
    new_dataset_root: str | Path,
    push_to_hub: bool = False,
) -> LeRobotDataset:
    """
    Downsample a LeRobotDataset to a target FPS by skipping frames.

    Args:
        original_dataset: The source dataset.
        target_fps: The target frames per second (must be a divisor of original fps).
        new_repo_id: Repository id for the new dataset.
        new_dataset_root: The root directory where the new dataset will be written.
        push_to_hub: Whether to push the new dataset to the hub.

    Returns:
        A new downsampled LeRobotDataset.

    Raises:
        ValueError: If target_fps is not a valid divisor of the original fps.
    """
    original_fps = original_dataset.fps

    if target_fps > original_fps:
        raise ValueError(
            f"target_fps ({target_fps}) cannot be greater than original fps ({original_fps})"
        )

    if original_fps % target_fps != 0:
        raise ValueError(
            f"target_fps ({target_fps}) must be a divisor of original fps ({original_fps})"
        )

    skip_factor = original_fps // target_fps
    logging.info(f"Downsampling from {original_fps} fps to {target_fps} fps (skip factor: {skip_factor})")

    # Create a new dataset with the target fps
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=target_fps,
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=original_dataset.meta.info["features"],
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    prev_episode_index = -1
    frame_in_episode = 0

    for frame_idx in tqdm(range(len(original_dataset)), desc="Downsampling dataset"):
        frame = original_dataset[frame_idx]
        current_episode = frame["episode_index"].item()

        # Check if we're starting a new episode
        if current_episode != prev_episode_index:
            # Save the previous episode if it exists
            if prev_episode_index != -1:
                new_dataset.save_episode()
            prev_episode_index = current_episode
            frame_in_episode = 0

        # Only keep frames at the target fps intervals
        if frame_in_episode % skip_factor == 0:
            # Create a copy of the frame to add to the new dataset
            new_frame = {}
            for key, value in frame.items():
                if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                    continue
                if key in (DONE, REWARD):
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        value = value.unsqueeze(0)
                if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.ndim == 0:
                    value = value.unsqueeze(0)
                new_frame[key] = value

            new_frame["task"] = frame["task"]
            new_dataset.add_frame(new_frame)

        frame_in_episode += 1

    # Save the last episode
    if prev_episode_index != -1:
        new_dataset.save_episode()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


def resize_dataset(
    original_dataset: LeRobotDataset,
    resize_size: tuple[int, int],
    new_repo_id: str,
    new_dataset_root: str | Path,
    image_keys: list[str] | None = None,
    push_to_hub: bool = False,
) -> LeRobotDataset:
    """
    Resize images in a LeRobotDataset to the specified dimensions.

    Args:
        original_dataset: The source dataset.
        resize_size: Target size as (height, width).
        new_repo_id: Repository id for the new dataset.
        new_dataset_root: The root directory where the new dataset will be written.
        image_keys: List of image keys to resize. If None, all image keys will be resized.
        push_to_hub: Whether to push the new dataset to the hub.

    Returns:
        A new LeRobotDataset with resized images.
    """
    logging.info(f"Resizing images to {resize_size}")

    # Determine which image keys to resize
    if image_keys is None:
        # Resize all camera/image keys
        image_keys = original_dataset.meta.camera_keys
    else:
        # Validate that specified keys exist
        for key in image_keys:
            if key not in original_dataset.meta.camera_keys:
                raise ValueError(f"Image key '{key}' not found in dataset")

    # Create a new dataset with updated image shapes
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=original_dataset.meta.info["features"],
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    # Update the metadata for resized image keys
    for key in image_keys:
        if key in new_dataset.meta.info["features"]:
            nb_channels = new_dataset.meta.info["features"][key]["shape"][0]
            new_dataset.meta.info["features"][key]["shape"] = [nb_channels, *resize_size]

    prev_episode_index = -1

    for frame_idx in tqdm(range(len(original_dataset)), desc="Resizing images"):
        frame = original_dataset[frame_idx]
        current_episode = frame["episode_index"].item()

        # Check if we're starting a new episode
        if current_episode != prev_episode_index:
            # Save the previous episode if it exists
            if prev_episode_index != -1:
                new_dataset.save_episode()
            prev_episode_index = current_episode

        # Create a copy of the frame to add to the new dataset
        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if key in (DONE, REWARD):
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    value = value.unsqueeze(0)

            # Resize if it's one of the specified image keys
            if key in image_keys:
                resized = F.resize(value, resize_size)
                value = resized.clamp(0, 1)

            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.ndim == 0:
                value = value.unsqueeze(0)

            new_frame[key] = value

        new_frame["task"] = frame["task"]
        new_dataset.add_frame(new_frame)

    # Save the last episode
    if prev_episode_index != -1:
        new_dataset.save_episode()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


def delete_episodes(
    original_dataset: LeRobotDataset,
    episodes_to_delete: list[int],
    new_repo_id: str,
    new_dataset_root: str | Path,
    push_to_hub: bool = False,
) -> LeRobotDataset:
    """
    Create a new dataset excluding specified episodes.

    Args:
        original_dataset: The source dataset.
        episodes_to_delete: List of episode indices to exclude from the new dataset.
        new_repo_id: Repository id for the new dataset.
        new_dataset_root: The root directory where the new dataset will be written.
        push_to_hub: Whether to push the new dataset to the hub.

    Returns:
        A new LeRobotDataset without the deleted episodes.
    """
    episodes_to_delete_set = set(episodes_to_delete)
    logging.info(f"Deleting episodes: {sorted(episodes_to_delete_set)}")

    # Validate episode indices
    for ep_idx in episodes_to_delete_set:
        if ep_idx < 0 or ep_idx >= original_dataset.meta.total_episodes:
            raise ValueError(
                f"Episode index {ep_idx} is out of range [0, {original_dataset.meta.total_episodes})"
            )

    # Create a new dataset
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=original_dataset.meta.info["features"],
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    prev_episode_index = -1
    episodes_kept = 0

    for frame_idx in tqdm(range(len(original_dataset)), desc="Deleting episodes"):
        frame = original_dataset[frame_idx]
        current_episode = frame["episode_index"].item()

        # Skip frames from episodes we want to delete
        if current_episode in episodes_to_delete_set:
            continue

        # Check if we're starting a new episode
        if current_episode != prev_episode_index:
            # Save the previous episode if it exists
            if prev_episode_index != -1:
                new_dataset.save_episode()
                episodes_kept += 1
            prev_episode_index = current_episode

        # Create a copy of the frame to add to the new dataset
        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if key in (DONE, REWARD):
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    value = value.unsqueeze(0)
            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.ndim == 0:
                value = value.unsqueeze(0)
            new_frame[key] = value

        new_frame["task"] = frame["task"]
        new_dataset.add_frame(new_frame)

    # Save the last episode
    if prev_episode_index != -1:
        new_dataset.save_episode()
        episodes_kept += 1

    logging.info(
        f"Kept {episodes_kept} episodes out of {original_dataset.meta.total_episodes} "
        f"(deleted {len(episodes_to_delete_set)} episodes)"
    )

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset
