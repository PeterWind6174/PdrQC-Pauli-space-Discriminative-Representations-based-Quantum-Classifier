# Small Processed Image Data

This lightweight PdrQC package includes only:

```text
mnist/processed/train.csv.gz
mnist/processed/test.csv.gz
mnist/processed/metadata.json
```

The subset contains 20 training and 10 test images per digit. It is supplied only to
verify that the code runs end to end; it is not the full paper benchmark.

To download and convert the complete MNIST and Fashion-MNIST IDX files, run from the
`code/` directory:

```bash
python dataset/download_and_prepare_image_datasets.py \
  --datasets mnist,fashion-mnist \
  --output dataset \
  --overwrite
```

Raw downloads are excluded by `.gitignore`.
