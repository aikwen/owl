# owl-imdl
![Python Version](https://img.shields.io/badge/python->=3.10-blue)
[![PyPI version](https://img.shields.io/pypi/v/owl-imdl.svg)](https://pypi.org/project/owl-imdl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install owl-imdl
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

### Initialize a Template

Run the following command in any directory:

```bash
owl init
```

This command allows you to select a built-in template and generate the corresponding script in the current directory.

### Check Version

```bash
owl version
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
