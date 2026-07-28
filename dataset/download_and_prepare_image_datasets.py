from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FileSpec:
    url: str
    path: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    files: tuple[FileSpec, ...]
    label_names: dict[str, str] | None = None


FASHION_LABELS = {
    "0": "T-shirt/top",
    "1": "Trouser",
    "2": "Pullover",
    "3": "Dress",
    "4": "Coat",
    "5": "Sandal",
    "6": "Shirt",
    "7": "Sneaker",
    "8": "Bag",
    "9": "Ankle boot",
}


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        files=(
            FileSpec(
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
                "mnist/raw/train-images-idx3-ubyte.gz",
            ),
            FileSpec(
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
                "mnist/raw/train-labels-idx1-ubyte.gz",
            ),
            FileSpec(
                "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
                "mnist/raw/t10k-images-idx3-ubyte.gz",
            ),
            FileSpec(
                "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
                "mnist/raw/t10k-labels-idx1-ubyte.gz",
            ),
        ),
    ),
    "fashion-mnist": DatasetSpec(
        name="fashion-mnist",
        files=(
            FileSpec(
                "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
                "fashion-mnist/raw/train-images-idx3-ubyte.gz",
            ),
            FileSpec(
                "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
                "fashion-mnist/raw/train-labels-idx1-ubyte.gz",
            ),
            FileSpec(
                "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
                "fashion-mnist/raw/t10k-images-idx3-ubyte.gz",
            ),
            FileSpec(
                "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
                "fashion-mnist/raw/t10k-labels-idx1-ubyte.gz",
            ),
        ),
        label_names=FASHION_LABELS,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw MNIST and Fashion-MNIST IDX files and convert them to the processed "
            "CSV.GZ format used by the PdrQC scripts."
        )
    )
    parser.add_argument(
        "--datasets",
        default="mnist,fashion-mnist",
        help="Comma-separated data sets to prepare: mnist, fashion-mnist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT,
        help="Output dataset root. Defaults to this dataset/ directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite processed CSV.GZ files even if they already exist.",
    )
    return parser.parse_args()


def build_opener() -> urllib.request.OpenerDirector:
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    return opener


def download_file(opener: urllib.request.OpenerDirector, root: Path, file_spec: FileSpec) -> Path:
    destination = root / file_spec.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"skip   {destination}")
        return destination

    print(f"fetch  {file_spec.url}")
    with opener.open(file_spec.url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    print(f"saved  {destination} ({destination.stat().st_size} bytes)")
    return destination


def gunzip_file(source: Path) -> Path:
    target = source.with_suffix("")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        print(f"skip   {target}")
        return target

    print(f"unzip  {source}")
    with gzip.open(source, "rb") as compressed, target.open("wb") as decompressed:
        shutil.copyfileobj(compressed, decompressed)
    return target


def read_idx_images(path: Path) -> tuple[int, int, int, bytes]:
    with path.open("rb") as file_obj:
        magic = int.from_bytes(file_obj.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Unexpected IDX image magic number in {path}: {magic}")
        count = int.from_bytes(file_obj.read(4), "big")
        rows = int.from_bytes(file_obj.read(4), "big")
        cols = int.from_bytes(file_obj.read(4), "big")
        payload = file_obj.read()

    expected = count * rows * cols
    if len(payload) != expected:
        raise ValueError(f"Unexpected IDX image payload size in {path}: {len(payload)} != {expected}")
    return count, rows, cols, payload


def read_idx_labels(path: Path) -> bytes:
    with path.open("rb") as file_obj:
        magic = int.from_bytes(file_obj.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label magic number in {path}: {magic}")
        count = int.from_bytes(file_obj.read(4), "big")
        payload = file_obj.read()

    if len(payload) != count:
        raise ValueError(f"Unexpected IDX label payload size in {path}: {len(payload)} != {count}")
    return payload


def write_image_csv_gz(
    images_path: Path,
    labels_path: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, int]:
    if output_path.exists() and not overwrite:
        print(f"skip   {output_path}")
        return {}

    count, rows, cols, image_payload = read_idx_images(images_path)
    label_payload = read_idx_labels(labels_path)
    if len(label_payload) != count:
        raise ValueError(f"Image/label count mismatch for {images_path} and {labels_path}")

    pixels_per_image = rows * cols
    headers = ["label", *[f"pixel_{index}" for index in range(pixels_per_image)]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="", compresslevel=1) as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(headers)
        for index in range(count):
            start = index * pixels_per_image
            stop = start + pixels_per_image
            writer.writerow([label_payload[index], *image_payload[start:stop]])

    print(f"saved  {output_path}")
    return {
        "num_items": count,
        "rows": rows,
        "cols": cols,
        "num_pixels": pixels_per_image,
    }


def prepare_dataset(root: Path, dataset: DatasetSpec, opener: urllib.request.OpenerDirector, overwrite: bool) -> None:
    print(f"\n== {dataset.name} ==")
    downloaded = [download_file(opener, root, file_spec) for file_spec in dataset.files]
    raw_paths = [gunzip_file(path) for path in downloaded]

    raw_dir = root / dataset.name / "raw"
    processed_dir = root / dataset.name / "processed"
    train_meta = write_image_csv_gz(
        raw_dir / "train-images-idx3-ubyte",
        raw_dir / "train-labels-idx1-ubyte",
        processed_dir / "train.csv.gz",
        overwrite=overwrite,
    )
    test_meta = write_image_csv_gz(
        raw_dir / "t10k-images-idx3-ubyte",
        raw_dir / "t10k-labels-idx1-ubyte",
        processed_dir / "test.csv.gz",
        overwrite=overwrite,
    )

    metadata = {
        "dataset": dataset.name,
        "raw_files": [str(path.relative_to(root)) for path in raw_paths],
        "processed_files": ["processed/train.csv.gz", "processed/test.csv.gz"],
        "format": "CSV.GZ with a label column followed by pixel_0 ... pixel_783",
    }
    if train_meta:
        metadata["train"] = train_meta
    if test_meta:
        metadata["test"] = test_meta
    if dataset.label_names is not None:
        metadata["label_names"] = dataset.label_names

    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved  {metadata_path}")


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}. Supported datasets: {sorted(DATASETS)}")

    opener = build_opener()
    for name in selected:
        prepare_dataset(root=root, dataset=DATASETS[name], opener=opener, overwrite=args.overwrite)

    print(f"\nCompleted. Processed files are under {root}")


if __name__ == "__main__":
    main()
