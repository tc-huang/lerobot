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

import pytest
import torch

from lerobot.datasets.preprocess_dataset import delete_episodes, downsample_dataset, resize_dataset


def test_downsample_dataset(tmp_path, lerobot_dataset_factory):
    """Test downsampling FPS of a dataset."""
    # Create a dataset with 30 fps and 2 episodes with 60 frames each
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original", total_episodes=2, total_frames=60, fps=30
    )

    # Downsample to 15 fps (skip factor of 2)
    new_dataset = downsample_dataset(
        original_dataset=original_dataset,
        target_fps=15,
        new_repo_id="test/downsampled",
        new_dataset_root=tmp_path / "downsampled",
        push_to_hub=False,
    )

    # Check fps is correct
    assert new_dataset.fps == 15

    # Check that the number of frames is approximately half
    # Each episode should have half the frames
    expected_frames = 60 // 2  # 30 frames total
    assert new_dataset.num_frames == expected_frames

    # Check that episodes are preserved
    assert new_dataset.num_episodes == 2


def test_downsample_dataset_invalid_fps(tmp_path, lerobot_dataset_factory):
    """Test that downsampling with invalid target fps raises an error."""
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original", total_episodes=1, total_frames=30, fps=30
    )

    # Test target_fps > original_fps
    with pytest.raises(ValueError, match="cannot be greater than original fps"):
        downsample_dataset(
            original_dataset=original_dataset,
            target_fps=60,
            new_repo_id="test/invalid",
            new_dataset_root=tmp_path / "invalid",
        )

    # Test target_fps not a divisor of original_fps
    with pytest.raises(ValueError, match="must be a divisor of original fps"):
        downsample_dataset(
            original_dataset=original_dataset,
            target_fps=20,
            new_repo_id="test/invalid",
            new_dataset_root=tmp_path / "invalid",
        )


def test_resize_dataset(tmp_path, lerobot_dataset_factory):
    """Test resizing images in a dataset."""
    # Create a dataset with images
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=2,
        total_frames=20,
    )

    # Get original image shape
    camera_keys = original_dataset.meta.camera_keys
    assert len(camera_keys) > 0, "Dataset should have camera keys for this test"

    # Resize to (64, 64)
    new_size = (64, 64)
    new_dataset = resize_dataset(
        original_dataset=original_dataset,
        resize_size=new_size,
        new_repo_id="test/resized",
        new_dataset_root=tmp_path / "resized",
        push_to_hub=False,
    )

    # Check that image shapes are updated
    for key in camera_keys:
        new_shape = new_dataset.meta.features[key]["shape"]
        assert new_shape[1:] == list(new_size), f"Expected {new_size}, got {new_shape[1:]}"

    # Check that the number of frames and episodes is preserved
    assert new_dataset.num_frames == original_dataset.num_frames
    assert new_dataset.num_episodes == original_dataset.num_episodes

    # Check that a frame has the correct image size
    frame = new_dataset[0]
    for key in camera_keys:
        assert frame[key].shape[1:] == torch.Size(new_size), f"Frame image shape mismatch for {key}"


def test_resize_dataset_specific_keys(tmp_path, lerobot_dataset_factory):
    """Test resizing only specific image keys."""
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=1,
        total_frames=10,
    )

    camera_keys = original_dataset.meta.camera_keys
    if len(camera_keys) < 1:
        pytest.skip("Need at least one camera key for this test")

    # Resize only the first camera key
    keys_to_resize = [camera_keys[0]]
    new_size = (64, 64)

    new_dataset = resize_dataset(
        original_dataset=original_dataset,
        resize_size=new_size,
        new_repo_id="test/resized_partial",
        new_dataset_root=tmp_path / "resized_partial",
        image_keys=keys_to_resize,
        push_to_hub=False,
    )

    # Check that only the specified key is resized
    assert new_dataset.meta.features[keys_to_resize[0]]["shape"][1:] == list(new_size)


def test_resize_dataset_invalid_key(tmp_path, lerobot_dataset_factory):
    """Test that resizing with invalid image key raises an error."""
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=1,
        total_frames=10,
    )

    with pytest.raises(ValueError, match="not found in dataset"):
        resize_dataset(
            original_dataset=original_dataset,
            resize_size=(64, 64),
            new_repo_id="test/invalid",
            new_dataset_root=tmp_path / "invalid",
            image_keys=["nonexistent_key"],
        )


def test_delete_episodes(tmp_path, lerobot_dataset_factory):
    """Test deleting specific episodes from a dataset."""
    # Create a dataset with 5 episodes
    total_episodes = 5
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=total_episodes,
        total_frames=100,
    )

    # Delete episodes 1 and 3
    episodes_to_delete = [1, 3]
    new_dataset = delete_episodes(
        original_dataset=original_dataset,
        episodes_to_delete=episodes_to_delete,
        new_repo_id="test/filtered",
        new_dataset_root=tmp_path / "filtered",
        push_to_hub=False,
    )

    # Check that the number of episodes is reduced
    expected_episodes = total_episodes - len(episodes_to_delete)
    assert new_dataset.num_episodes == expected_episodes

    # Check that frames from deleted episodes are not in the new dataset
    new_episode_indices = set()
    for i in range(len(new_dataset)):
        ep_idx = new_dataset.hf_dataset[i]["episode_index"].item()
        new_episode_indices.add(ep_idx)

    # Episodes are renumbered starting from 0
    # Original episodes 0, 2, 4 should map to new episodes 0, 1, 2
    assert len(new_episode_indices) == expected_episodes


def test_delete_episodes_invalid_index(tmp_path, lerobot_dataset_factory):
    """Test that deleting episodes with invalid indices raises an error."""
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=3,
        total_frames=30,
    )

    # Test negative index
    with pytest.raises(ValueError, match="out of range"):
        delete_episodes(
            original_dataset=original_dataset,
            episodes_to_delete=[-1],
            new_repo_id="test/invalid",
            new_dataset_root=tmp_path / "invalid",
        )

    # Test index >= total_episodes
    with pytest.raises(ValueError, match="out of range"):
        delete_episodes(
            original_dataset=original_dataset,
            episodes_to_delete=[10],
            new_repo_id="test/invalid",
            new_dataset_root=tmp_path / "invalid",
        )


def test_delete_all_episodes_except_one(tmp_path, lerobot_dataset_factory):
    """Test deleting all episodes except one."""
    total_episodes = 3
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=total_episodes,
        total_frames=30,
    )

    # Delete all episodes except episode 1
    episodes_to_delete = [0, 2]
    new_dataset = delete_episodes(
        original_dataset=original_dataset,
        episodes_to_delete=episodes_to_delete,
        new_repo_id="test/single_episode",
        new_dataset_root=tmp_path / "single_episode",
        push_to_hub=False,
    )

    # Should have only 1 episode
    assert new_dataset.num_episodes == 1


def test_downsample_preserves_features(tmp_path, lerobot_dataset_factory):
    """Test that downsampling preserves all features."""
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=1,
        total_frames=30,
        fps=30,
    )

    new_dataset = downsample_dataset(
        original_dataset=original_dataset,
        target_fps=15,
        new_repo_id="test/downsampled",
        new_dataset_root=tmp_path / "downsampled",
    )

    # Check that all non-metadata features are preserved
    original_features = set(original_dataset.features.keys())
    new_features = set(new_dataset.features.keys())

    assert original_features == new_features


def test_resize_preserves_non_image_features(tmp_path, lerobot_dataset_factory):
    """Test that resizing preserves non-image features."""
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=1,
        total_frames=10,
    )

    new_dataset = resize_dataset(
        original_dataset=original_dataset,
        resize_size=(64, 64),
        new_repo_id="test/resized",
        new_dataset_root=tmp_path / "resized",
    )

    # Check that non-image features are preserved
    frame = new_dataset[0]
    original_frame = original_dataset[0]

    for key in original_frame:
        if key not in original_dataset.meta.camera_keys and key != "task":
            # Non-image features should be present
            assert key in frame


def test_combined_operations(tmp_path, lerobot_dataset_factory):
    """Test combining multiple preprocessing operations."""
    # Create original dataset
    original_dataset = lerobot_dataset_factory(
        root=tmp_path / "original",
        total_episodes=4,
        total_frames=120,
        fps=30,
    )

    # First, delete some episodes
    after_delete = delete_episodes(
        original_dataset=original_dataset,
        episodes_to_delete=[0, 3],
        new_repo_id="test/step1",
        new_dataset_root=tmp_path / "step1",
    )

    # Then, downsample FPS
    after_downsample = downsample_dataset(
        original_dataset=after_delete,
        target_fps=15,
        new_repo_id="test/step2",
        new_dataset_root=tmp_path / "step2",
    )

    # Finally, resize images
    final_dataset = resize_dataset(
        original_dataset=after_downsample,
        resize_size=(64, 64),
        new_repo_id="test/final",
        new_dataset_root=tmp_path / "final",
    )

    # Verify the final dataset has all transformations applied
    assert final_dataset.num_episodes == 2  # Deleted 2 out of 4
    assert final_dataset.fps == 15  # Downsampled from 30 to 15

    # Check image size
    if len(final_dataset.meta.camera_keys) > 0:
        key = final_dataset.meta.camera_keys[0]
        assert final_dataset.meta.features[key]["shape"][1:] == [64, 64]
