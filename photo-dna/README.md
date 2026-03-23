[![Crates.io Version](https://img.shields.io/crates/v/photo-dna?style=for-the-badge)](https://crates.io/crates/photo-dna)
[![docs.rs](https://img.shields.io/docsrs/photo-dna?style=for-the-badge)](https://docs.rs/photo-dna/latest/photo_dna/)
![GitHub License](https://img.shields.io/github/license/ail-project/photo-dna?style=for-the-badge)

<!-- cargo-rdme start -->

PhotoDNA - A Rust implementation of the PhotoDNA perceptual hashing algorithm

This crate provides a pure Rust implementation of PhotoDNA, a perceptual hashing algorithm
originally developed by Microsoft for detecting similar images. This implementation is a
port of the work by [ArcaneNibble](https://github.com/ArcaneNibble/open-alleged-photodna)
and provides a safe, efficient way to generate and compare image hashes.

## Overview

PhotoDNA creates compact, robust hash representations of images that can be used to:
- Detect similar or identical images
- Find duplicates in image collections
- Implement content-based image retrieval
- Build content moderation systems

The algorithm is designed to be resistant to common image modifications like resizing,
compression, color adjustments, and minor content changes, while still detecting
perceptually similar images.

## Key Features

- **Multiple input formats**: Create hashes from image files, raw pixels, or in-memory data
- **Similarity metrics**: Euclidean and log2p distance metrics with normalized similarity scores
- **Serialization support**: Hex, base64, and binary representations

## Feature Flags

- **`base64`**: Enable base64 encoding/decoding support
- **`serde`**: Enable Serde serialization/deserialization support

## Basic Usage

```rust
use photo_dna::Hash;

// Create hashes from image files
let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
let hash2 = Hash::from_image_path("tests/image_2.jpg").unwrap();

// Calculate similarity
let similarity = hash1.similarity_log2p(&hash2);
println!("Image similarity: {:.2}%", similarity * 100.0);
```

## Distance Metrics

The crate provides two distance metrics:

- **Euclidean Distance**: Standard geometric distance in multi-dimensional space
- **Log2p Distance**: Logarithmic distance that emphasizes small perceptual differences

Both metrics provide corresponding similarity scores normalized to the 0.0-1.0 range.

## Examples

### Creating hashes from different sources

```rust
use photo_dna::Hash;
use image::DynamicImage;

// From an image file
let hash = Hash::from_image_path("tests/random.png").unwrap();

// From a DynamicImage
let img = image::open("tests/random.png").unwrap();
let hash = Hash::from_image(&img).unwrap();

// From raw RGB pixels
let width = 100;
let height = 100;
let pixels = vec![[255u8, 0, 0]; (width * height) as usize];
let hash = Hash::from_rgb_pixels(width, height, pixels).unwrap();
```

### Comparing images

```rust
use photo_dna::Hash;

let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
let hash2 = Hash::from_image_path("tests/image_2.jpg").unwrap();

// Using Euclidean distance
let euclidean_dist = hash1.distance_euclid(&hash2);
let euclidean_sim = hash1.similarity_euclidian(&hash2);

// Using log2p distance (often more perceptually relevant)
let log2p_dist = hash1.distance_log2p(&hash2);
let log2p_sim = hash1.similarity_log2p(&hash2);
```

### Serialization

```rust
use photo_dna::Hash;

let hash = Hash::from_image_path("tests/image_1.jpg").unwrap();

// Convert to hex string
let hex_string = hash.to_hex_string();

// Convert back from hex
let restored_hash = Hash::from_hex_str(&hex_string).unwrap();
assert_eq!(hash, restored_hash);
```

## License

This project is licensed under the GPLv3 License.

<!-- cargo-rdme end -->
