#!/usr/bin/env python3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetDict, concatenate_datasets, load_dataset
from PIL import Image

# =========================
# Config (edit if needed)
# =========================

DATASET_ID = "glab-caltech/TWIN"
OUTPUT_DIR = Path("training/data")
IMAGES_DIR = OUTPUT_DIR / "images"  # all images live directly here
TEST_SIZE = 5000
SEED = 42
BATCH_SIZE = 1000
MAX_IMAGE_SIDE = 512  # largest side after "resizing" in metadata


# =========================
# Helpers
# =========================

def load_hf_dataset(dataset_id: str):
    dataset = load_dataset(dataset_id)
    if isinstance(dataset, DatasetDict):
        splits = [dataset[name] for name in sorted(dataset.keys())]
        return concatenate_datasets(splits)
    return dataset


def get_local_image_path(rel_path: str) -> Path:
    """
    Map dataset path ('images/xxx.jpg' or 'images_00001/xxx.jpg') -> local file in IMAGES_DIR.
    We only use the filename; all files are in training/data/images.
    """
    filename = Path(rel_path).name
    return IMAGES_DIR / filename


def get_resized_size(
    image_path: Path,
    size_cache: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    """
    Return (resized_height, resized_width) such that the largest side is at most MAX_IMAGE_SIDE,
    using the same logic as the old make_image_entry() helper.
    """
    key = str(image_path)
    cached = size_cache.get(key)
    if cached:
        return cached

    with Image.open(image_path) as img:
        width, height = img.size

    max_side = max(width, height)
    if max_side <= MAX_IMAGE_SIDE:
        target_width, target_height = width, height
    else:
        scale = MAX_IMAGE_SIDE / max_side
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))

    size_cache[key] = (target_height, target_width)
    return target_height, target_width


def build_record(
    example: dict,
    index: int,
    size_cache: dict[str, tuple[int, int]],
) -> dict:
    question = example["question"]
    answer = example["answer"]
    image_1 = example["image_1"]
    image_2 = example["image_2"]

    local_1 = get_local_image_path(image_1)
    local_2 = get_local_image_path(image_2)
    h1, w1 = get_resized_size(local_1, size_cache)
    h2, w2 = get_resized_size(local_2, size_cache)

    prompt = f"<image>\n<image>\n\n{question}"
    image_files = [Path(image_1).name, Path(image_2).name]
    pair_id = f"{Path(image_1).stem}_{Path(image_2).stem}"

    return {
        "data_source": DATASET_ID,
        "prompt": [{"content": prompt, "role": "user"}],
        "images": [
            {"image": str(local_1), "resized_height": h1, "resized_width": w1},
            {"image": str(local_2), "resized_height": h2, "resized_width": w2},
        ],
        "ability": "vision",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "image_files": image_files,
            "index": index,
            "pair_id": pair_id,
        },
    }


def write_parquet(dataset, output_path: Path, start_index: int):
    size_cache: dict[str, tuple[int, int]] = {}
    writer = None
    batch = []

    for offset, example in enumerate(dataset):
        record = build_record(
            example,
            index=start_index + offset,
            size_cache=size_cache,
        )
        batch.append(record)
        if len(batch) >= BATCH_SIZE:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()


# =========================
# Main preprocessing
# =========================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_hf_dataset(DATASET_ID)
    total = len(dataset)
    dataset = dataset.shuffle(seed=SEED)

    test_dataset = dataset.select(range(TEST_SIZE))
    train_dataset = dataset.select(range(TEST_SIZE, total))

    train_path = OUTPUT_DIR / "train.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

    # Overwrite any existing files
    if train_path.exists():
        train_path.unlink()
    if test_path.exists():
        test_path.unlink()

    write_parquet(
        test_dataset,
        output_path=test_path,
        start_index=0,
    )
    write_parquet(
        train_dataset,
        output_path=train_path,
        start_index=TEST_SIZE,
    )

    print(f"Wrote {len(train_dataset)} train rows to {train_path}")
    print(f"Wrote {len(test_dataset)} test rows to {test_path}")


if __name__ == "__main__":
    main()
