[![PyPI - Version](https://img.shields.io/pypi/v/photo-dna-rs?style=for-the-badge)](https://pypi.org/project/photo-dna-rs/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/photo-dna-rs?style=for-the-badge)
![GitHub License](https://img.shields.io/github/license/ail-project/photo-dna?style=for-the-badge)

<!-- cargo-rdme start -->

# photo-dna-rs

A high-performance Python module for generating and comparing PhotoDNA image hashes.

## Overview

**photo-dna-rs** provides Python bindings to a Rust implementation of the PhotoDNA image hashing algorithm.

## Installation

### From PyPi

```bash
uv add photo-dna-rs
```

### From source

```bash
# 1. Clone repository and navigate to python directory
git clone https://github.com/ail-project/photo-dna
cd photo-dna/python

# 2. Install
pip install -e .

# 3. Explore APIs
python -c "from photo_dna_rs import Hash; help(Hash)"
```

## Quick Start

```python
import photo_dna_rs

# Create hash from image file
hash1 = photo_dna_rs.Hash.from_image_path("image1.jpg")
hash2 = photo_dna_rs.Hash.from_image_path("image2.jpg")

# Calculate similarity
similarity = hash1.similarity_log2p(hash2)
print(f"Images are {similarity*100:.1f}% similar")
```

## Usage

### Creating Hashes

```python
# From image file
hash = photo_dna_rs.Hash.from_image_path("photo.jpg")

# From RGB pixels
width = 100
height = 100
pixels = [[255, 0, 0] for _ in range(width * height)]
hash = photo_dna_rs.Hash.from_rgb_pixels(width, height, pixels)

# From hex string
hex_str = "deadbeef" * 36  # 288 characters
hash = photo_dna_rs.Hash.from_hex_str(hex_str)

# From bytes
hash_bytes = bytes(144)
hash = photo_dna_rs.Hash.from_bytes(hash_bytes)
```

### Comparing Hashes

```python
# Euclidean distance and similarity
distance = hash1.distance_euclidian(hash2)
similarity = hash1.similarity_euclidian(hash2)

# Log2p distance and similarity (recommended)
distance = hash1.distance_log2p(hash2)
similarity = hash1.similarity_log2p(hash2)
```

## License

This project is licensed under the GPLv3 License.

<!-- cargo-rdme end -->
