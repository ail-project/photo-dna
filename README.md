# PhotoDNA: Perceptual Image Hashing in Rust and Python

A pure Rust implementation and Python bindings of the PhotoDNA perceptual hashing algorithm for image similarity detection.

## Overview

**PhotoDNA** provides robust image hashing and comparison capabilities with:

- **Pure Rust implementation** for maximum performance and safety
- **Python bindings** via PyO3 for easy integration
- **Perceptual hashing** that detects similar images even after transformations
- **Multiple distance metrics** for flexible similarity comparison

## Packages

### Rust Crate: [`photo-dna`](./photo-dna/README.md)

The core implementation in pure Rust.

```bash
cargo add photo-dna
```

### Python Package: [`photo-dna-rs`](./python/README.md)

Python bindings using PyO3.

```bash
pip install photo-dna-rs
```

## 🐍 Quick Start (Python)

```python
from photo_dna_rs import Hash

# Create hashes from images
hash1 = Hash.from_image_path("image1.jpg")
hash2 = Hash.from_image_path("image2.jpg")

# Calculate similarity (0.0 = different, 1.0 = identical)
similarity = hash1.similarity_log2p(hash2)
print(f"Images are {similarity*100:.1f}% similar")

# Convert between formats
hex_string = hash1.to_hex_str()
hash_from_hex = Hash.from_hex_str(hex_string)
```

## 🦀 Quick Start (Rust)

```rust
use photo_dna::Hash;

// Create hashes from image files
let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
let hash2 = Hash::from_image_path("tests/image_2.jpg").unwrap();

// Calculate similarity using log2p metric (recommended)
let similarity = hash1.similarity_log2p(&hash2);
println!("Image similarity: {:.2}%", similarity * 100.0);

// Serialize to hex string
let hex_string = hash1.to_hex_string();
let restored_hash = Hash::from_hex_str(&hex_string).unwrap();
```

## Development

### Building

```bash
# Build both crates
cargo build --all

# Build Python package
cd python
maturin develop
```

### Testing

```bash
# Test Rust crate
cargo test --all-features

# Test Python package
cd python
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a new branch
2. **Write tests** for new functionality
3. **Update documentation** for any changes
4. **Run tests**: `cargo test` and `pytest tests/ -v`
5. **Format code**: Use `rustfmt` for Rust code

## 🙏 Acknowledgments

This implementation is based on the excellent work by **ArcaneNibble** in the [Open Alleged PhotoDNA](https://github.com/ArcaneNibble/open-alleged-photodna) project. The original reference implementation provided the foundation for this Rust port.

## Support

- **GitHub Issues**: Report bugs and request features
- **Pull Requests**: Contribute improvements

## License

This project is licensed under **GPLv3**. See the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by the [AIL Project](https://github.com/ail-project) for robust image analysis and content moderation.*
