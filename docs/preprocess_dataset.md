# Dataset Preprocessing Utilities

LeRobot provides utilities to preprocess existing datasets before policy training. These utilities allow you to:

- **Downsample FPS**: Reduce the frames per second of a dataset
- **Resize images**: Change the resolution of images in a dataset
- **Delete episodes**: Remove specific episodes from a dataset

## Usage

### Command Line Interface

The preprocessing utilities are accessible via the `lerobot_preprocess_dataset.py` script.

#### Downsample FPS

Reduce the FPS of a dataset by keeping only every Nth frame:

```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/pusht \
    --operation=downsample \
    --target-fps=15 \
    --new-repo-id=lerobot/pusht_15fps
```

**Note**: The target FPS must be a divisor of the original FPS (e.g., if original is 30 fps, valid targets are 30, 15, 10, 6, 5, 3, 2, 1).

#### Resize Images

Change the resolution of all images in a dataset:

```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/pusht \
    --operation=resize \
    --resize-height=128 \
    --resize-width=128 \
    --new-repo-id=lerobot/pusht_128x128
```

To resize only specific camera views:

```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/aloha_mobile \
    --operation=resize \
    --resize-height=256 \
    --resize-width=256 \
    --image-keys="observation.images.cam_high,observation.images.cam_left_wrist" \
    --new-repo-id=lerobot/aloha_mobile_256x256
```

#### Delete Episodes

Remove specific episodes from a dataset:

```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/pusht \
    --operation=delete \
    --episodes-to-delete="0,5,10" \
    --new-repo-id=lerobot/pusht_filtered
```

#### Additional Options

- `--root`: Specify a custom root directory for the source dataset
- `--new-root`: Specify a custom root directory for the output dataset
- `--push-to-hub`: Push the preprocessed dataset to the Hugging Face Hub

### Python API

You can also use the preprocessing utilities programmatically:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.preprocess_dataset import (
    downsample_dataset,
    resize_dataset,
    delete_episodes,
)

# Load original dataset
original_dataset = LeRobotDataset("lerobot/pusht")

# Downsample FPS
downsampled = downsample_dataset(
    original_dataset=original_dataset,
    target_fps=15,
    new_repo_id="lerobot/pusht_15fps",
    new_dataset_root="/path/to/output",
    push_to_hub=False,
)

# Resize images
resized = resize_dataset(
    original_dataset=original_dataset,
    resize_size=(128, 128),
    new_repo_id="lerobot/pusht_128x128",
    new_dataset_root="/path/to/output",
    image_keys=None,  # None means all image keys
    push_to_hub=False,
)

# Delete episodes
filtered = delete_episodes(
    original_dataset=original_dataset,
    episodes_to_delete=[0, 5, 10],
    new_repo_id="lerobot/pusht_filtered",
    new_dataset_root="/path/to/output",
    push_to_hub=False,
)
```

## Combining Operations

You can chain multiple preprocessing operations together:

```python
# First delete unwanted episodes
step1 = delete_episodes(
    original_dataset=original_dataset,
    episodes_to_delete=[0, 1, 2],
    new_repo_id="temp/step1",
    new_dataset_root="/tmp/step1",
)

# Then downsample FPS
step2 = downsample_dataset(
    original_dataset=step1,
    target_fps=15,
    new_repo_id="temp/step2",
    new_dataset_root="/tmp/step2",
)

# Finally resize images
final_dataset = resize_dataset(
    original_dataset=step2,
    resize_size=(128, 128),
    new_repo_id="lerobot/pusht_preprocessed",
    new_dataset_root="/path/to/final",
    push_to_hub=True,
)
```

## Important Notes

1. **Non-destructive**: All preprocessing operations create a new dataset and do not modify the original.

2. **Storage**: Preprocessing creates a complete copy of the dataset, so ensure you have sufficient disk space.

3. **Video encoding**: If the original dataset uses videos, the preprocessed dataset will also use videos and will need to re-encode them.

4. **Metadata preservation**: All dataset metadata (stats, tasks, robot type, etc.) are preserved and updated appropriately.

5. **FPS constraints**: When downsampling, the target FPS must evenly divide the original FPS to maintain temporal consistency.

## Use Cases

### Training Speed Optimization

Downsampling FPS can significantly speed up training while potentially maintaining performance:

```bash
# Original dataset: 30 fps
# Downsampled: 10 fps (3x fewer frames)
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/aloha_sim \
    --operation=downsample \
    --target-fps=10 \
    --new-repo-id=lerobot/aloha_sim_10fps
```

### Memory Optimization

Resizing images can reduce memory usage during training:

```bash
# Original images: 640x480
# Resized: 224x224 (much smaller)
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/aloha_mobile \
    --operation=resize \
    --resize-height=224 \
    --resize-width=224 \
    --new-repo-id=lerobot/aloha_mobile_224
```

### Data Cleaning

Remove low-quality or failed episodes:

```bash
python src/lerobot/scripts/lerobot_preprocess_dataset.py \
    --repo-id=lerobot/my_dataset \
    --operation=delete \
    --episodes-to-delete="3,7,12,15" \
    --new-repo-id=lerobot/my_dataset_cleaned
```
