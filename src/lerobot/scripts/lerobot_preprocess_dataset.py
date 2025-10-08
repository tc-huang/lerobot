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
CLI tool for preprocessing LeRobot datasets.

Examples:

Downsample a dataset FPS:
```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/pusht \
    --operation=downsample \
    --target-fps=15 \
    --new-repo-id=lerobot/pusht_15fps
```

Resize images in a dataset:
```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/pusht \
    --operation=resize \
    --resize-height=128 \
    --resize-width=128 \
    --new-repo-id=lerobot/pusht_128x128
```

Delete specific episodes:
```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/pusht \
    --operation=delete \
    --episodes-to-delete="0,5,10" \
    --new-repo-id=lerobot/pusht_filtered
```
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.preprocess_dataset import delete_episodes, downsample_dataset, resize_dataset
from lerobot.utils.utils import init_logging


def main():
    parser = argparse.ArgumentParser(description="Preprocess a LeRobot dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The repository id of the LeRobot dataset to process.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="The root directory of the LeRobot dataset.",
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=["downsample", "resize", "delete"],
        help="The preprocessing operation to perform.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        required=True,
        help="The repository id for the new preprocessed dataset.",
    )
    parser.add_argument(
        "--new-root",
        type=str,
        default=None,
        help="The root directory for the new dataset. If not provided, defaults to the new-repo-id under LEROBOT_HOME.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the new dataset to the hub.",
    )

    # Downsample options
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Target FPS for downsampling (required for 'downsample' operation).",
    )

    # Resize options
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Target height for resizing images (required for 'resize' operation).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Target width for resizing images (required for 'resize' operation).",
    )
    parser.add_argument(
        "--image-keys",
        type=str,
        default=None,
        help="Comma-separated list of image keys to resize. If not provided, all image keys will be resized.",
    )

    # Delete options
    parser.add_argument(
        "--episodes-to-delete",
        type=str,
        default=None,
        help="Comma-separated list of episode indices to delete (required for 'delete' operation).",
    )

    args = parser.parse_args()

    init_logging()

    # Load the original dataset
    logging.info(f"Loading dataset from {args.repo_id}")
    original_dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    # Perform the requested operation
    if args.operation == "downsample":
        if args.target_fps is None:
            raise ValueError("--target-fps is required for 'downsample' operation")

        new_dataset = downsample_dataset(
            original_dataset=original_dataset,
            target_fps=args.target_fps,
            new_repo_id=args.new_repo_id,
            new_dataset_root=args.new_root or args.new_repo_id,
            push_to_hub=args.push_to_hub,
        )
        logging.info(f"Downsampled dataset saved to {new_dataset.root}")

    elif args.operation == "resize":
        if args.resize_height is None or args.resize_width is None:
            raise ValueError("--resize-height and --resize-width are required for 'resize' operation")

        image_keys = None
        if args.image_keys is not None:
            image_keys = [key.strip() for key in args.image_keys.split(",")]

        new_dataset = resize_dataset(
            original_dataset=original_dataset,
            resize_size=(args.resize_height, args.resize_width),
            new_repo_id=args.new_repo_id,
            new_dataset_root=args.new_root or args.new_repo_id,
            image_keys=image_keys,
            push_to_hub=args.push_to_hub,
        )
        logging.info(f"Resized dataset saved to {new_dataset.root}")

    elif args.operation == "delete":
        if args.episodes_to_delete is None:
            raise ValueError("--episodes-to-delete is required for 'delete' operation")

        episodes_to_delete = [int(ep.strip()) for ep in args.episodes_to_delete.split(",")]

        new_dataset = delete_episodes(
            original_dataset=original_dataset,
            episodes_to_delete=episodes_to_delete,
            new_repo_id=args.new_repo_id,
            new_dataset_root=args.new_root or args.new_repo_id,
            push_to_hub=args.push_to_hub,
        )
        logging.info(f"Filtered dataset saved to {new_dataset.root}")

    logging.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
