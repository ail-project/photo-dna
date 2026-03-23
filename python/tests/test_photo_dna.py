"""
Test suite for photo_dna_rs Python module.
Tests all the PyO3 bindings for the PhotoDNA hash functionality.
"""

import os
import tempfile
from pathlib import Path
from random import randint

import pytest
from PIL import Image

import photo_dna_rs
from photo_dna_rs import Hash

# Test image paths (relative to the main project root)
TEST_IMAGES_DIR = Path(__file__).parent.parent.parent / "photo-dna" / "tests"
IMAGE_1 = TEST_IMAGES_DIR / "image_1.jpg"
IMAGE_2 = TEST_IMAGES_DIR / "image_2.jpg"
RANDOM_IMAGE = TEST_IMAGES_DIR / "random.png"


def test_from_image_path():
    """Test creating hash from image file path."""
    # Test with existing image
    hash1 = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    assert hash1 is not None

    # Test that hash has correct byte length
    bytes_repr = hash1.as_bytes()
    assert len(bytes_repr) == 144

    # Test with different image
    hash2 = photo_dna_rs.Hash.from_image_path(str(IMAGE_2))
    assert hash2 is not None

    # Hashes should be different
    assert hash1.to_hex_str() != hash2.to_hex_str()


def test_from_image_path_invalid():
    """Test error handling for invalid image paths."""
    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_image_path("nonexistent_image.jpg")


def test_from_rgb_pixels():
    """Test creating hash from RGB pixel data."""
    width = 128
    height = 64
    pixels = [
        (randint(0, 255), randint(0, 255), randint(0, 255))
        for _ in range(width * height)
    ]

    hash = photo_dna_rs.Hash.from_rgb_pixels(width, height, pixels)
    assert hash is not None

    # Test byte representation
    bytes_repr = hash.as_bytes()
    assert len(bytes_repr) == 144

    # Test with different pixel data
    pixels2 = [
        (randint(0, 255), randint(0, 255), randint(0, 255))
        for _ in range(width * height)
    ]
    hash2 = photo_dna_rs.Hash.from_rgb_pixels(width, height, pixels2)

    # Hashes should be different
    assert hash.to_hex_str() != hash2.to_hex_str()


def test_from_rgb_pixels_invalid():
    """Test error handling for invalid pixel data."""
    # Test with zero width/height
    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_rgb_pixels(0, 1, [(255, 0, 0)])

    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_rgb_pixels(1, 0, [(255, 0, 0)])


def test_from_hex_str():
    """Test creating hash from hexadecimal string."""
    # Create a hash from an image first
    original_hash = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    hex_str = original_hash.to_hex_str()

    # Verify hex string length
    assert len(hex_str) == 288  # 144 bytes * 2 characters per byte

    # Create hash from hex string
    hash_from_hex = photo_dna_rs.Hash.from_hex_str(hex_str)

    # Should be identical
    assert hash_from_hex.to_hex_str() == original_hash.to_hex_str()
    assert hash_from_hex.as_bytes() == original_hash.as_bytes()


def test_from_hex_str_invalid():
    """Test error handling for invalid hex strings."""
    # Test with wrong length
    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_hex_str("deadbeef")  # Too short

    # Test with invalid characters
    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_hex_str("g" * 288)  # Invalid hex characters


def test_from_bytes():
    """Test creating hash from bytes."""
    # Create a hash from an image first
    original_hash = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    bytes_data = original_hash.as_bytes()

    # Verify byte length
    assert len(bytes_data) == 144

    # Create hash from bytes
    hash_from_bytes = photo_dna_rs.Hash.from_bytes(bytes_data)

    # Should be identical
    assert hash_from_bytes.to_hex_str() == original_hash.to_hex_str()
    assert hash_from_bytes.as_bytes() == original_hash.as_bytes()


def test_from_bytes_invalid():
    """Test error handling for invalid byte data."""
    # Test with wrong length
    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_bytes(bytes(100))  # Too short

    with pytest.raises(ValueError):
        photo_dna_rs.Hash.from_bytes(bytes(200))  # Too long


def test_as_bytes():
    """Test getting raw bytes from hash."""
    hash = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    bytes_repr = hash.as_bytes()

    assert isinstance(bytes_repr, bytes)
    assert len(bytes_repr) == 144


def test_to_hex_str():
    """Test converting hash to hexadecimal string."""
    hash = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    hex_str = hash.to_hex_str()

    assert isinstance(hex_str, str)
    assert len(hex_str) == 288

    # Should only contain hex characters
    assert all(c in "0123456789abcdef" for c in hex_str)


def test_distance_euclidian():
    """Test Euclidean distance calculation."""
    hash1 = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    hash2 = photo_dna_rs.Hash.from_image_path(str(IMAGE_2))
    hash3 = photo_dna_rs.Hash.from_image_path(str(RANDOM_IMAGE))

    # Distance should be non-negative
    distance1 = hash1.distance_euclidian(hash2)
    assert distance1 >= 0.0

    # Distance to self should be 0
    distance_self = hash1.distance_euclidian(hash1)
    assert distance_self == 0.0

    # Different images should have different distances
    distance2 = hash1.distance_euclidian(hash3)
    assert distance1 != distance2


def test_similarity_euclidian():
    """Test Euclidean similarity calculation."""
    hash1 = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    hash2 = photo_dna_rs.Hash.from_image_path(str(IMAGE_2))

    # Similarity should be between 0 and 1
    similarity = hash1.similarity_euclidian(hash2)
    assert 0.0 <= similarity <= 1.0

    # Similarity to self should be 1.0
    similarity_self = hash1.similarity_euclidian(hash1)
    assert similarity_self == 1.0


def test_distance_log2p():
    """Test log2p distance calculation."""
    hash1 = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    hash2 = photo_dna_rs.Hash.from_image_path(str(IMAGE_2))

    # Distance should be non-negative
    distance = hash1.distance_log2p(hash2)
    assert distance >= 0.0

    # Distance to self should be 0
    distance_self = hash1.distance_log2p(hash1)
    assert distance_self == 0.0


def test_similarity_log2p():
    """Test log2p similarity calculation."""
    hash1 = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))
    hash2 = photo_dna_rs.Hash.from_image_path(str(IMAGE_2))

    # Similarity should be between 0 and 1
    similarity = hash1.similarity_log2p(hash2)
    assert 0.0 <= similarity <= 1.0

    # Similarity to self should be 1.0
    similarity_self = hash1.similarity_log2p(hash1)
    assert similarity_self == 1.0


def test_consistency_across_methods():
    """Test that different creation methods produce consistent results."""
    # Create hash from image
    hash_from_image = photo_dna_rs.Hash.from_image_path(str(IMAGE_1))

    # Create hash from hex string of the same image
    hex_str = hash_from_image.to_hex_str()
    hash_from_hex = photo_dna_rs.Hash.from_hex_str(hex_str)

    # Create hash from bytes of the same image
    bytes_data = hash_from_image.as_bytes()
    hash_from_bytes = photo_dna_rs.Hash.from_bytes(bytes_data)

    # All should be identical
    assert hash_from_image.to_hex_str() == hash_from_hex.to_hex_str()
    assert hash_from_image.to_hex_str() == hash_from_bytes.to_hex_str()
    assert hash_from_image.as_bytes() == hash_from_hex.as_bytes()
    assert hash_from_image.as_bytes() == hash_from_bytes.as_bytes()


def test_with_pil_image():
    """Test integration with PIL Image objects."""
    # Create a PIL image
    pil_image = Image.new("RGB", (100, 100), color=(255, 0, 0))  # Red image

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        pil_image.save(temp_path)

    try:
        # Create hash from the saved image
        hash = photo_dna_rs.Hash.from_image_path(temp_path)
        assert hash is not None
        assert len(hash.as_bytes()) == 144
    finally:
        # Clean up
        os.unlink(temp_path)

    hash_from_pixels = Hash.from_rgb_pixels(
        pil_image.width,
        pil_image.height,
        list(pil_image.get_flattened_data()),  # ty:ignore[invalid-argument-type]
    )

    assert hash_from_pixels.as_bytes() == hash.as_bytes()
    assert hash_from_pixels == hash


def test_different_image_sizes():
    """Test that different image sizes produce valid hashes."""
    test_cases = [
        (10, 10),  # Small image
        (100, 100),  # Medium image
        (500, 300),  # Larger image
    ]

    for width, height in test_cases:
        # Create a uniform color image
        pixels = [(128, 128, 128) for _ in range(width * height)]  # Gray image
        hash = photo_dna_rs.Hash.from_rgb_pixels(width, height, pixels)

        assert hash is not None
        assert len(hash.as_bytes()) == 144


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
