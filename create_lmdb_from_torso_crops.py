import io
import os
import csv
import argparse

import lmdb
import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        img = Image.open(io.BytesIO(imageBin)).convert("RGB")
        return np.prod(img.size) > 0
    except Exception:
        return False


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_lmdb_dataset(input_path, csv_file, output_path, checkValid=True):
    """
    Create LMDB dataset from CSV file
    Args:
        input_path: Root directory of image files
        csv_file: Path to the annotation CSV file
        output_path: Path to the output LMDB directory
        checkValid: Whether to check the validity of images
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)

    cache = {}
    cnt = 1
    valid_samples = 0

    print(f"Reading CSV file {csv_file}...")
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total_rows = len(rows)

        for i, row in enumerate(rows):
            rel_path = row["rel_path"]
            jersey_number = row["jersey_number_new"]
            is_visible = row["jersey_number_is_visible"]
            is_legible = row["is_legible"]

            # 背番号が見えるかつ読めるデータのみ使用
            if (
                is_visible.lower() == "true"
                and is_legible.lower() == "true"
                and jersey_number
            ):
                image_path = os.path.join(input_path, rel_path)
                try:
                    with open(image_path, "rb") as f:
                        imageBin = f.read()
                except FileNotFoundError:
                    print(f"Warning: Image file {image_path} not found")
                    continue

                if checkValid:
                    try:
                        img = Image.open(io.BytesIO(imageBin)).convert("RGB")
                    except Exception as e:
                        with open(output_path + "/error_image_log.txt", "a") as log:
                            log.write(f"{i}-th image data error: {image_path}, {e}\n")
                        continue
                    if np.prod(img.size) == 0:
                        print(f"{image_path} is not a valid image")
                        continue

                imageKey = f"image-{cnt:09d}".encode()
                labelKey = f"label-{cnt:09d}".encode()
                cache[imageKey] = imageBin
                cache[labelKey] = jersey_number.encode()

                if cnt % 100 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print(
                        f"Progress: {cnt} / {total_rows} (Valid samples: {valid_samples})"
                    )
                cnt += 1
                valid_samples += 1

    # 残りのキャッシュを書き込む
    cache["num-samples".encode()] = str(valid_samples).encode()
    writeCache(env, cache)
    env.close()
    print(f"LMDB dataset created. Valid samples: {valid_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LMDB dataset from CSV")
    parser.add_argument(
        "--train_images",
        required=True,
        help="Root directory of training image files",
    )
    parser.add_argument(
        "--train_csv_file",
        required=True,
        help="Path to the training annotation CSV file",
    )
    parser.add_argument(
        "--test_images",
        required=True,
        help="Root directory of test image files",
    )
    parser.add_argument(
        "--test_csv_file",
        required=True,
        help="Path to the test annotation CSV file",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Root directory for the output LMDB datasets",
    )
    parser.add_argument(
        "--no_check_valid",
        action="store_false",
        dest="check_valid",
        help="Skip image validity check",
    )

    args = parser.parse_args()

    # 出力ディレクトリを作成
    train_output_path = os.path.join(args.output_dir, "train")
    val_output_path = os.path.join(args.output_dir, "val")
    test_output_path = os.path.join(args.output_dir, "test")

    # 訓練データセットの作成
    print(f"Creating training dataset...")
    create_lmdb_dataset(
        args.train_images, args.train_csv_file, train_output_path, args.check_valid
    )

    # テストデータセットの作成
    print(f"Creating test dataset...")
    create_lmdb_dataset(
        args.test_images, args.test_csv_file, test_output_path, args.check_valid
    )

    # バリデーションデータをテストデータからコピー
    print(f"Creating validation dataset from test data...")
    os.makedirs(val_output_path, exist_ok=True)
    os.system(f"cp {test_output_path}/data.mdb {val_output_path}/")
    os.system(f"cp {test_output_path}/lock.mdb {val_output_path}/")
    print(f"Created validation dataset at: {val_output_path}")

    print(f"All datasets created successfully:")
    print(f"  - Training: {train_output_path}")
    print(f"  - Validation: {val_output_path}")
    print(f"  - Test: {test_output_path}")
