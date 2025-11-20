# owl-imdl
![Python Version](https://img.shields.io/badge/python->3.7-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

my IMDL utils.

## Installation (TestPyPI)

Currently, this package is available on **TestPyPI**. You can install it using the following command:

```bash
pip install -i https://test.pypi.org/simple/ owl-imdl
```

## Manual Dependencies

To keep the package lightweight and flexible, this project does not enforce deep learning framework dependencies (to avoid version conflicts).

Please manually install the following dependencies according to your environment before use:

1. **PyTorch**: Visit [pytorch](https://pytorch.org/get-started/locally/) to get the command for your CUDA version.
2. Other Essentials:
```bash
pip install numpy Pillow albumentations
```

## Quick Start

### Initialize a Project

Run the following command in any directory to generate a training script (e.g., `my_project.py`):

```bash
owl init my_project
```

## Dataset Structure

```text
my_dataset/
├── gt/                 # Ground Truth images
├── tp/                 # Tampered/Target images
└── my_dataset.json     # Index file (MUST match the folder name!)
```

JSON Format (my_dataset.json):

```json
[
  {
    "tp": "tampered_image_01.jpg",
    "gt": "mask_01.png"
  },
  {
    "tp": "tampered_image_02.jpg",
    "gt": "mask_02.png"
  }
]
```