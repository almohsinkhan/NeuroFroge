#!/usr/bin/env python3
"""Download and convert CIFAR-10 binary batches to NeuroFroge CSV files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import tarfile
from typing import Iterable
from urllib.request import urlopen


ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXTRACTED_DIR = "cifar-10-batches-py"
IMAGE_SIZE = 3 * 32 * 32
RECORD_SIZE = 1 + IMAGE_SIZE
TRAIN_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))
TEST_BATCHES = ("test_batch",)
TRAIN_COUNT = 50_000
TEST_COUNT = 10_000


class Cifar10Error(RuntimeError):
    """Raised when a CIFAR-10 archive or generated dataset is invalid."""


def download_archive(archive_path: Path) -> None:
    """Download the archive to archive_path, unless it already exists."""
    if archive_path.is_file():
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    try:
        with urlopen(ARCHIVE_URL, timeout=60) as response, \
                temporary_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        os.replace(temporary_path, archive_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise Cifar10Error(
            f"Failed to download CIFAR-10 archive: {error}"
        ) from error


def safe_extract(archive_path: Path, destination: Path) -> Path:
    """Extract a tar archive while rejecting path traversal and links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination / EXTRACTED_DIR
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            destination_resolved = destination.resolve()
            for member in archive.getmembers():
                member_path = (destination / member.name).resolve()
                if os.path.commonpath(
                    (str(destination_resolved), str(member_path))
                ) != str(destination_resolved):
                    raise Cifar10Error(
                        f"Archive member escapes extraction directory: "
                        f"{member.name}"
                    )
                if member.issym() or member.islnk() or not (
                    member.isdir() or member.isfile()
                ):
                    raise Cifar10Error(
                        f"Unsupported archive member: {member.name}"
                    )
            archive.extractall(destination)
    except (tarfile.TarError, OSError) as error:
        raise Cifar10Error(
            f"Failed to extract archive {archive_path}: {error}"
        ) from error
    if not root.is_dir():
        raise Cifar10Error(
            f"Archive does not contain expected directory {EXTRACTED_DIR}"
        )
    return root


def read_batch(path: Path) -> Iterable[tuple[int, bytes]]:
    """Yield validated ``(label, channel_first_pixels)`` records."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise Cifar10Error(f"Failed to read batch {path}: {error}") from error
    if not data or len(data) % RECORD_SIZE != 0:
        raise Cifar10Error(
            f"Corrupted batch {path}: size is not a multiple of {RECORD_SIZE}"
        )
    for offset in range(0, len(data), RECORD_SIZE):
        label = data[offset]
        pixels = data[offset + 1:offset + RECORD_SIZE]
        if not 0 <= label <= 9 or len(pixels) != IMAGE_SIZE:
            raise Cifar10Error(f"Invalid record in batch {path}")
        yield label, pixels


def write_split(
    source_directory: Path,
    batch_names: tuple[str, ...],
    image_path: Path,
    label_path: Path,
) -> int:
    """Convert binary batches to paired image and label CSV files."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with image_path.open("w", newline="") as images, \
            label_path.open("w", newline="") as labels:
        image_writer = csv.writer(images, lineterminator="\n")
        label_writer = csv.writer(labels, lineterminator="\n")
        for batch_name in batch_names:
            batch_path = source_directory / batch_name
            if not batch_path.is_file():
                raise Cifar10Error(f"Missing CIFAR-10 batch: {batch_path}")
            for label, pixels in read_batch(batch_path):
                image_writer.writerow(pixels)
                label_writer.writerow((label,))
                count += 1
    return count


def validate_csv(image_path: Path, label_path: Path, expected_count: int) -> None:
    """Validate generated CSV counts, shapes, labels, and pixel ranges."""
    image_count = 0
    label_count = 0
    with image_path.open(newline="") as images, \
            label_path.open(newline="") as labels:
        image_reader = csv.reader(images)
        label_reader = csv.reader(labels)
        for image_row, label_row in zip(image_reader, label_reader):
            if len(image_row) != IMAGE_SIZE or len(label_row) != 1:
                raise Cifar10Error(
                    f"Invalid generated row in {image_path}"
                )
            try:
                pixels = [int(value) for value in image_row]
                label = int(label_row[0])
            except ValueError as error:
                raise Cifar10Error(
                    f"Non-integer generated value in {image_path}"
                ) from error
            if any(pixel < 0 or pixel > 255 for pixel in pixels):
                raise Cifar10Error(f"Pixel outside 0-255 in {image_path}")
            if not 0 <= label <= 9:
                raise Cifar10Error(f"Label outside 0-9 in {label_path}")
            image_count += 1
            label_count += 1
        if next(image_reader, None) is not None or \
                next(label_reader, None) is not None:
            raise Cifar10Error("Image and label CSV lengths do not match")
    if image_count != expected_count or label_count != expected_count:
        raise Cifar10Error(
            f"Expected {expected_count} samples, got "
            f"{image_count} images and {label_count} labels"
        )


def prepare(
    output_directory: Path,
    archive_path: Path | None = None,
    extracted_directory: Path | None = None,
) -> None:
    """Prepare train and test CIFAR-10 CSV files."""
    archive_path = archive_path or output_directory / ARCHIVE_NAME
    extracted_directory = extracted_directory or output_directory
    download_archive(archive_path)
    source_directory = safe_extract(archive_path, extracted_directory)

    train_images = output_directory / "cifar10_train_images.csv"
    train_labels = output_directory / "cifar10_train_labels.csv"
    test_images = output_directory / "cifar10_test_images.csv"
    test_labels = output_directory / "cifar10_test_labels.csv"
    train_count = write_split(
        source_directory, TRAIN_BATCHES, train_images, train_labels
    )
    test_count = write_split(
        source_directory, TEST_BATCHES, test_images, test_labels
    )
    if train_count != TRAIN_COUNT or test_count != TEST_COUNT:
        raise Cifar10Error(
            f"Unexpected CIFAR-10 sample counts: "
            f"{train_count} training and {test_count} test"
        )
    validate_csv(train_images, train_labels, TRAIN_COUNT)
    validate_csv(test_images, test_labels, TEST_COUNT)
    print(f"Generated {train_count} training samples and {test_count} test samples")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10 binary batches to NeuroFroge CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cifar10"),
        help="Directory for the archive, extraction, and generated CSV files",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Existing archive path, or download destination",
    )
    args = parser.parse_args()
    try:
        prepare(args.output_dir, args.archive)
    except Cifar10Error as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
