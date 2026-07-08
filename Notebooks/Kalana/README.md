# Baseline Segmentation

This folder contains the simple CNN segmentation baseline in `main.py`.

## Dataset path setup

At the top of `main.py`, these variables control where the script looks for the dataset:

```python
IMAGE_FOLDER = "images"
MASK_FOLDER = "masks"
```

### Recommended setup

Keep the dataset folders inside this same `Kalana` directory:

```text
Kalana/
  main.py
  README.md
  images/
  masks/
```

With this structure, the default values work without any changes.

### If your dataset is elsewhere

Update the variables to point to the correct paths.

Examples:

```python
IMAGE_FOLDER = r"C:\path\to\your\images"
MASK_FOLDER = r"C:\path\to\your\masks"
```

or, if the folders are in a parent directory:

```python
IMAGE_FOLDER = r"..\images"
MASK_FOLDER = r"..\masks"
```

## Important note

The script uses relative paths based on the folder you run it from. If you run `main.py` from another directory, make sure `IMAGE_FOLDER` and `MASK_FOLDER` still point to valid locations.

## Filename convention

The images use names like:

- `242583_sat_08.jpg`

The matching masks use names like:

- `242583_mask_08.jpg`

So the image and mask names must follow the same number pattern.
