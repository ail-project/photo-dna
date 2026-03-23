#![deny(unused_imports)]

//! # photo-dna-rs
//!
//! A high-performance Python module for generating and comparing PhotoDNA image hashes.
//!
//! ## Overview
//!
//! **photo-dna-rs** provides Python bindings to a Rust implementation of the PhotoDNA image hashing algorithm.
//!
//! ## Installation
//!
//! ### From PyPi
//!
//! ```bash
//! uv add photo-dna-rs
//! ```
//!
//! ### From source
//!
//! ```bash
//! # 1. Clone repository and navigate to python directory
//! git clone https://github.com/ail-project/photo-dna
//! cd photo-dna/python
//!
//! # 2. Install
//! pip install -e .
//!
//! # 3. Explore APIs
//! python -c "from photo_dna_rs import Hash; help(Hash)"
//! ```
//!
//! ## Quick Start
//!
//! ```python
//! import photo_dna_rs
//!
//! # Create hash from image file
//! hash1 = photo_dna_rs.Hash.from_image_path("image1.jpg")
//! hash2 = photo_dna_rs.Hash.from_image_path("image2.jpg")
//!
//! # Calculate similarity
//! similarity = hash1.similarity_log2p(hash2)
//! print(f"Images are {similarity*100:.1f}% similar")
//! ```
//!
//! ## Usage
//!
//! ### Creating Hashes
//!
//! ```python
//! # From image file
//! hash = photo_dna_rs.Hash.from_image_path("photo.jpg")
//!
//! # From RGB pixels
//! width = 100
//! height = 100
//! pixels = [[255, 0, 0] for _ in range(width * height)]
//! hash = photo_dna_rs.Hash.from_rgb_pixels(width, height, pixels)
//!
//! # From hex string
//! hex_str = "deadbeef" * 36  # 288 characters
//! hash = photo_dna_rs.Hash.from_hex_str(hex_str)
//!
//! # From bytes
//! hash_bytes = bytes(144)
//! hash = photo_dna_rs.Hash.from_bytes(hash_bytes)
//! ```
//!
//! ### Comparing Hashes
//!
//! ```python
//! # Euclidean distance and similarity
//! distance = hash1.distance_euclidian(hash2)
//! similarity = hash1.similarity_euclidian(hash2)
//!
//! # Log2p distance and similarity (recommended)
//! distance = hash1.distance_log2p(hash2)
//! similarity = hash1.similarity_log2p(hash2)
//! ```
//!
//! ## License
//!
//! This project is licensed under the GPLv3 License.

use std::path::PathBuf;

use pyo3::{exceptions::PyValueError, prelude::*};

/// Python class for PhotoDNA hash operations.
///
/// This struct is designed for Python usage and should be documented
/// through the Python docstrings, not Rust documentation.
#[doc(hidden)]
#[pyclass]
pub struct Hash(photo_dna::Hash);

#[doc(hidden)]
impl From<photo_dna::Hash> for Hash {
    fn from(value: photo_dna::Hash) -> Self {
        Self(value)
    }
}

#[doc(hidden)]
fn to_pyvalue_err(e: photo_dna::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[doc(hidden)]
#[pymethods]
impl Hash {
    /// Create a PhotoDNA hash from RGB pixel data.
    ///
    /// Args:
    ///     width (int): Image width in pixels
    ///     height (int): Image height in pixels
    ///     pixels (List[List[int]]): List of RGB pixels as [R, G, B] arrays (0-255 each)
    ///
    /// Returns:
    ///     Hash: PhotoDNA hash object
    ///
    /// Raises:
    ///     ValueError: If pixel data cannot be converted to a valid image
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> # Create a 2x2 red image
    ///     >>> width = 2
    ///     >>> height = 2
    ///     >>> pixels = [[255, 0, 0]] * (width * height)
    ///     >>> hash = photo_dna_rs.Hash.from_rgb_pixels(width, height, pixels)
    #[staticmethod]
    pub fn from_rgb_pixels(width: u32, height: u32, b: Vec<[u8; 3]>) -> PyResult<Self> {
        photo_dna::Hash::from_rgb_pixels(width, height, b).map_err(to_pyvalue_err).map(Hash::from)
    }

    /// Create a PhotoDNA hash from an image file.
    ///
    /// Args:
    ///     path (str or Path): Path to image file (JPEG, PNG, etc.)
    ///
    /// Returns:
    ///     Hash: PhotoDNA hash object
    ///
    /// Raises:
    ///     ValueError: If image cannot be processed
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash = photo_dna_rs.Hash.from_image_path("image.jpg")
    #[staticmethod]
    pub fn from_image_path(p: PathBuf) -> PyResult<Self> {
        photo_dna::Hash::from_image_path(p).map_err(to_pyvalue_err).map(Hash::from)
    }

    /// Create a PhotoDNA hash from a hexadecimal string.
    ///
    /// Args:
    ///     hex_string (str): Hexadecimal string (288 characters = 144 bytes)
    ///
    /// Returns:
    ///     Hash: PhotoDNA hash object
    ///
    /// Raises:
    ///     ValueError: If string length is incorrect or contains invalid hex characters
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hex_str = "deadbeef" * 36  # 288 character hex string
    ///     >>> hash = photo_dna_rs.Hash.from_hex_str(hex_str)
    #[staticmethod]
    pub fn from_hex_str(s: String) -> PyResult<Self> {
        photo_dna::Hash::from_hex_str(&s).map_err(to_pyvalue_err).map(Hash::from)
    }

    /// Create a PhotoDNA hash from bytes.
    ///
    /// Args:
    ///     byte_data (bytes): Bytes object containing hash data (must be 144 bytes)
    ///
    /// Returns:
    ///     Hash: PhotoDNA hash object
    ///
    /// Raises:
    ///     ValueError: If byte length is not exactly 144 bytes
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash_bytes = bytes(144)  # 144 zero bytes
    ///     >>> hash = photo_dna_rs.Hash.from_bytes(hash_bytes)
    #[staticmethod]
    pub fn from_bytes(b: &[u8]) -> PyResult<Self> {
        photo_dna::Hash::from_bytes(b).map_err(to_pyvalue_err).map(Hash::from)
    }

    /// Get the raw bytes of the hash.
    ///
    /// Returns:
    ///     bytes: Raw hash bytes (144 bytes)
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash = photo_dna_rs.Hash.from_image_path("image.jpg")
    ///     >>> raw_bytes = hash.as_bytes()
    ///     >>> len(raw_bytes)  # 144
    pub fn as_bytes(&self) -> &photo_dna::HashBuf {
        self.0.as_bytes()
    }

    /// Convert hash to hexadecimal string.
    ///
    /// Returns:
    ///     str: Hexadecimal string representation (288 characters)
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash = photo_dna_rs.Hash.from_image_path("image.jpg")
    ///     >>> hex_str = hash.to_hex_str()
    ///     >>> len(hex_str)  # 288
    pub fn to_hex_str(&self) -> String {
        self.0.to_hex_string()
    }

    /// Calculate Euclidean distance between two hashes.
    ///
    /// Args:
    ///     other (Hash): Another PhotoDNA hash to compare with
    ///
    /// Returns:
    ///     float: Euclidean distance (0.0 = identical, higher = more different)
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash1 = photo_dna_rs.Hash.from_image_path("image1.jpg")
    ///     >>> hash2 = photo_dna_rs.Hash.from_image_path("image2.jpg")
    ///     >>> distance = hash1.distance_euclidian(hash2)
    ///     >>> print(f"Distance: {distance:.2f}")
    pub fn distance_euclidian(&self, other: &Self) -> f64 {
        self.0.distance_euclid(&other.0)
    }

    /// Calculate similarity using Euclidean distance.
    ///
    /// Args:
    ///     other (Hash): Another PhotoDNA hash to compare with
    ///
    /// Returns:
    ///     float: Similarity score [0.0; 1.0]
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash1 = photo_dna_rs.Hash.from_image_path("image1.jpg")
    ///     >>> hash2 = photo_dna_rs.Hash.from_image_path("image2.jpg")
    ///     >>> similarity = hash1.similarity_euclidian(hash2)
    ///     >>> print(f"Similarity: {similarity*100:.1f}%")
    pub fn similarity_euclidian(&self, other: &Self) -> f64 {
        self.0.similarity_euclidian(&other.0)
    }

    /// Calculate log2p distance between two hashes.
    ///
    /// Args:
    ///     other (Hash): Another PhotoDNA hash to compare with
    ///
    /// Returns:
    ///     float: Log2p distance (0.0 = identical, higher = more different)
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash1 = photo_dna_rs.Hash.from_image_path("image1.jpg")
    ///     >>> hash2 = photo_dna_rs.Hash.from_image_path("image2.jpg")
    ///     >>> distance = hash1.distance_log2p(hash2)
    ///     >>> print(f"Log2p distance: {distance:.2f}")
    pub fn distance_log2p(&self, other: &Self) -> f64 {
        self.0.distance_log2p(&other.0)
    }

    /// Calculate similarity using log2p distance.
    ///
    /// Args:
    ///     other (Hash): Another PhotoDNA hash to compare with
    ///
    /// Returns:
    ///     float: Similarity score [0.0; 1.0]
    ///
    /// Example:
    ///     >>> import photo_dna_rs
    ///     >>> hash1 = photo_dna_rs.Hash.from_image_path("image1.jpg")
    ///     >>> hash2 = photo_dna_rs.Hash.from_image_path("image2.jpg")
    ///     >>> similarity = hash1.similarity_log2p(hash2)
    ///     >>> print(f"Similarity: {similarity*100:.1f}%")
    pub fn similarity_log2p(&self, other: &Self) -> f64 {
        self.0.similarity_log2p(&other.0)
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// Python module for PhotoDNA image hashing.
///
/// This module is designed for Python usage and should be documented
/// through the Python docstrings, not Rust documentation.
#[doc(hidden)]
#[pymodule]
mod photo_dna_rs {
    #[pymodule_export]
    use super::Hash;
}
