#!/usr/bin/env python3
"""Synthetic validation test for prepare_cifar10.py."""

from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import prepare_cifar10


class PrepareCifar10Test(unittest.TestCase):
    def test_conversion_and_rgb_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / prepare_cifar10.EXTRACTED_DIR
            source.mkdir()
            record = bytes([7]) + (
                bytes([10]) * 1024 +
                bytes([20]) * 1024 +
                bytes([30]) * 1024
            )
            (source / "data_batch_1").write_bytes(record)
            for name in prepare_cifar10.TRAIN_BATCHES[1:]:
                (source / name).write_bytes(record)
            (source / "test_batch").write_bytes(record)

            output = root / "output"
            train_images = output / "train_images.csv"
            train_labels = output / "train_labels.csv"
            count = prepare_cifar10.write_split(
                source,
                prepare_cifar10.TRAIN_BATCHES,
                train_images,
                train_labels,
            )
            self.assertEqual(count, 5)
            prepare_cifar10.validate_csv(train_images, train_labels, 5)
            first_row = train_images.read_text().splitlines()[0].split(",")
            self.assertEqual(len(first_row), 3072)
            self.assertEqual(first_row[:4], ["10"] * 4)
            self.assertEqual(first_row[1024:1028], ["20"] * 4)
            self.assertEqual(first_row[2048:2052], ["30"] * 4)

    def test_safe_extraction_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bad.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("../outside")
                info.size = 1
                import io
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(prepare_cifar10.Cifar10Error):
                prepare_cifar10.safe_extract(archive_path, root / "extract")


if __name__ == "__main__":
    unittest.main()
