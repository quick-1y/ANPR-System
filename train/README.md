# Training models

This directory contains lightweight pipelines for training the detector (YOLO)
and the OCR recognizer. Both trainers are configured through YAML files in
`train/train_config` and have simple CLI wrappers under `train/scripts`.

## YOLO detector

The YOLO trainer wraps the Ultralytics API and expects a dataset YAML describing
the train/val/test splits. The provided `train/train_config/detection_dataset.yaml`
assumes images live under `data/detection` with `images/train`, `images/val`, and
`images/test` subfolders.

Run training with:

```bash
python -m train.scripts.train_yolo --config train/train_config/yolo_train_config.yaml
```

Key options are stored in the config file (model checkpoint, batch size, image
size, patience, etc.). To point to a different dataset YAML or override values,
edit the config file or add CLI overrides supported by the trainer.

## OCR recognizer

The OCR pipeline trains a compact CRNN model using CTC loss. It expects a
`labels.csv` file with `image_path,label` rows (paths are relative to
`dataset_root` from the config) and grayscale images sized to the configured
height/width.

Start OCR training with:

```bash
python -m train.scripts.train_ocr --config train/train_config/ocr_config.yaml
```

The OCR config controls vocabulary, image size, batch size, learning rate, and
checkpointing. Checkpoints are written to the directory in `checkpoint_dir`, and
training can resume by setting `resume_from` to a saved `.pt` file.

## Available configuration files

- `train/train_config/yolo_train_config.yaml` – primary settings for YOLO
  training.
- `train/train_config/detection_dataset.yaml` – dataset description used by the
  YOLO config.
- `train/train_config/ocr_config.yaml` – settings for OCR training.

Adjust these files as needed for your datasets, hardware, and experiment names.
