#![deny(unused_imports)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! PhotoDNA - A Rust implementation of the PhotoDNA perceptual hashing algorithm
//!
//! This crate provides a pure Rust implementation of PhotoDNA, a perceptual hashing algorithm
//! originally developed by Microsoft for detecting similar images. This implementation is a
//! port of the work by [ArcaneNibble](https://github.com/ArcaneNibble/open-alleged-photodna)
//! and provides a safe, efficient way to generate and compare image hashes.
//!
//! ## Overview
//!
//! PhotoDNA creates compact, robust hash representations of images that can be used to:
//! - Detect similar or identical images
//! - Find duplicates in image collections
//! - Implement content-based image retrieval
//! - Build content moderation systems
//!
//! The algorithm is designed to be resistant to common image modifications like resizing,
//! compression, color adjustments, and minor content changes, while still detecting
//! perceptually similar images.
//!
//! ## Key Features
//!
//! - **Multiple input formats**: Create hashes from image files, raw pixels, or in-memory data
//! - **Similarity metrics**: Euclidean and log2p distance metrics with normalized similarity scores
//! - **Serialization support**: Hex, base64, and binary representations
//!
//! ## Feature Flags
//!
//! - **`base64`**: Enable base64 encoding/decoding support
//! - **`serde`**: Enable Serde serialization/deserialization support
//!
//! ## Basic Usage
//!
//! ```rust
//! use photo_dna::Hash;
//!
//! // Create hashes from image files
//! let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
//! let hash2 = Hash::from_image_path("tests/image_2.jpg").unwrap();
//!
//! // Calculate similarity
//! let similarity = hash1.similarity_log2p(&hash2);
//! println!("Image similarity: {:.2}%", similarity * 100.0);
//! ```
//!
//! ## Distance Metrics
//!
//! The crate provides two distance metrics:
//!
//! - **Euclidean Distance**: Standard geometric distance in multi-dimensional space
//! - **Log2p Distance**: Logarithmic distance that emphasizes small perceptual differences
//!
//! Both metrics provide corresponding similarity scores normalized to the 0.0-1.0 range.
//!
//! ## Examples
//!
//! ### Creating hashes from different sources
//!
//! ```rust
//! use photo_dna::Hash;
//! use image::DynamicImage;
//!
//! // From an image file
//! let hash = Hash::from_image_path("tests/random.png").unwrap();
//!
//! // From a DynamicImage
//! let img = image::open("tests/random.png").unwrap();
//! let hash = Hash::from_image(&img).unwrap();
//!
//! // From raw RGB pixels
//! let width = 100;
//! let height = 100;
//! let pixels = vec![[255u8, 0, 0]; (width * height) as usize];
//! let hash = Hash::from_rgb_pixels(width, height, pixels).unwrap();
//! ```
//!
//! ### Comparing images
//!
//! ```rust
//! use photo_dna::Hash;
//!
//! let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
//! let hash2 = Hash::from_image_path("tests/image_2.jpg").unwrap();
//!
//! // Using Euclidean distance
//! let euclidean_dist = hash1.distance_euclid(&hash2);
//! let euclidean_sim = hash1.similarity_euclidian(&hash2);
//!
//! // Using log2p distance (often more perceptually relevant)
//! let log2p_dist = hash1.distance_log2p(&hash2);
//! let log2p_sim = hash1.similarity_log2p(&hash2);
//! ```
//!
//! ### Serialization
//!
//! ```rust
//! use photo_dna::Hash;
//!
//! let hash = Hash::from_image_path("tests/image_1.jpg").unwrap();
//!
//! // Convert to hex string
//! let hex_string = hash.to_hex_string();
//!
//! // Convert back from hex
//! let restored_hash = Hash::from_hex_str(&hex_string).unwrap();
//! assert_eq!(hash, restored_hash);
//! ```
//!
//! ## License
//!
//! This project is licensed under the GPLv3 License.

use std::path::Path;

use image::{DynamicImage, ImageError, RgbImage};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "base64")]
use base64::{DecodeError, prelude::*};

// pre-computed log2p(x) rounded to 4
// so log2p(x) ~ LOG2P_TABLE[x]
// using pre-computed values makes log2p based distance
// computation significantly faster ~ 10x
#[allow(clippy::approx_constant)]
const LOG2P_TABLE: [f64; 256] = [
    0.0, 1.0, 1.585, 2.0, 2.3219, 2.585, 2.8074, 3.0, 3.1699, 3.3219, 3.4594, 3.585, 3.7004, 3.8074, 3.9069, 4.0, 4.0875, 4.1699, 4.2479, 4.3219,
    4.3923, 4.4594, 4.5236, 4.585, 4.6439, 4.7004, 4.7549, 4.8074, 4.858, 4.9069, 4.9542, 5.0, 5.0444, 5.0875, 5.1293, 5.1699, 5.2095, 5.2479,
    5.2854, 5.3219, 5.3576, 5.3923, 5.4263, 5.4594, 5.4919, 5.5236, 5.5546, 5.585, 5.6147, 5.6439, 5.6724, 5.7004, 5.7279, 5.7549, 5.7814, 5.8074,
    5.8329, 5.858, 5.8826, 5.9069, 5.9307, 5.9542, 5.9773, 6.0, 6.0224, 6.0444, 6.0661, 6.0875, 6.1085, 6.1293, 6.1497, 6.1699, 6.1898, 6.2095,
    6.2288, 6.2479, 6.2668, 6.2854, 6.3038, 6.3219, 6.3399, 6.3576, 6.375, 6.3923, 6.4094, 6.4263, 6.4429, 6.4594, 6.4757, 6.4919, 6.5078, 6.5236,
    6.5392, 6.5546, 6.5699, 6.585, 6.5999, 6.6147, 6.6294, 6.6439, 6.6582, 6.6724, 6.6865, 6.7004, 6.7142, 6.7279, 6.7415, 6.7549, 6.7682, 6.7814,
    6.7944, 6.8074, 6.8202, 6.8329, 6.8455, 6.858, 6.8704, 6.8826, 6.8948, 6.9069, 6.9189, 6.9307, 6.9425, 6.9542, 6.9658, 6.9773, 6.9887, 7.0,
    7.0112, 7.0224, 7.0334, 7.0444, 7.0553, 7.0661, 7.0768, 7.0875, 7.098, 7.1085, 7.1189, 7.1293, 7.1396, 7.1497, 7.1599, 7.1699, 7.1799, 7.1898,
    7.1997, 7.2095, 7.2192, 7.2288, 7.2384, 7.2479, 7.2574, 7.2668, 7.2761, 7.2854, 7.2946, 7.3038, 7.3129, 7.3219, 7.3309, 7.3399, 7.3487, 7.3576,
    7.3663, 7.375, 7.3837, 7.3923, 7.4009, 7.4094, 7.4179, 7.4263, 7.4346, 7.4429, 7.4512, 7.4594, 7.4676, 7.4757, 7.4838, 7.4919, 7.4998, 7.5078,
    7.5157, 7.5236, 7.5314, 7.5392, 7.5469, 7.5546, 7.5622, 7.5699, 7.5774, 7.585, 7.5925, 7.5999, 7.6073, 7.6147, 7.6221, 7.6294, 7.6366, 7.6439,
    7.6511, 7.6582, 7.6653, 7.6724, 7.6795, 7.6865, 7.6935, 7.7004, 7.7074, 7.7142, 7.7211, 7.7279, 7.7347, 7.7415, 7.7482, 7.7549, 7.7616, 7.7682,
    7.7748, 7.7814, 7.7879, 7.7944, 7.8009, 7.8074, 7.8138, 7.8202, 7.8265, 7.8329, 7.8392, 7.8455, 7.8517, 7.858, 7.8642, 7.8704, 7.8765, 7.8826,
    7.8887, 7.8948, 7.9009, 7.9069, 7.9129, 7.9189, 7.9248, 7.9307, 7.9366, 7.9425, 7.9484, 7.9542, 7.96, 7.9658, 7.9715, 7.9773, 7.983, 7.9887,
    7.9944, 8.0,
];

// Approximate log2(1 + x) for distance computation
#[inline(always)]
const fn alog2p(i: u8) -> f64 {
    LOG2P_TABLE[i as usize]
}

// Extracted constants
// These constants are used as weights for each differently-sized
// rectangle during the feature extraction phase.
// This is used in Equation 11 in the paper.
const WEIGHT_R1: f64 = 0.196967309312945;
const WEIGHT_R2: f64 = 0.448111736620497;
const WEIGHT_R3: f64 = 0.278037300453194;

// This is used for initial hash scaling.
// This is described in section 3.4 of the paper.
const HASH_SCALE_CONST: f64 = 1.03008177008737;

// This parameter is used to switch between "robust" and "short"
// hashes. It is not clear how exactly this is intended to be used
// (e.g. "short" hashes have a totally different postprocessing step).
// The only value used in practice is 6. Changing it may or may not work.
const GRID_SIZE_HYPERPARAMETER: usize = 6;
const GRADIENT_GRID_SIZE: usize = GRID_SIZE_HYPERPARAMETER * GRID_SIZE_HYPERPARAMETER * 4;

type FeatureGrid = [f64; FEATURE_GRID_SIZE];
type GradientGrid = [f64; GRADIENT_GRID_SIZE];

#[inline(always)]
fn preprocess_pixel_sum(im: RgbImage) -> Vec<u64> {
    let mut sum = Vec::with_capacity((im.width() * im.height()) as usize);

    let mut row_sum = 0;
    (0..im.width()).for_each(|x| {
        let p = im.get_pixel(x, 0);
        row_sum += p[0] as u64 + p[1] as u64 + p[2] as u64;
        sum.push(row_sum);
    });

    (1..im.height()).for_each(|y| {
        let mut row_sum = 0;
        (0..im.width()).for_each(|x| {
            let p = im.get_pixel(x, y);
            let last_row_sum = sum[((y - 1) * im.width() + x) as usize];
            row_sum += p[0] as u64 + p[1] as u64 + p[2] as u64;
            sum.push(row_sum + last_row_sum);
        })
    });

    sum
}

/// # ----- (3.2) Feature extraction -----
///
/// # This is equal to 26. This means that the `u` and `v` coordinates
/// # mentioned in the paper both range from [0, 25].
const FEATURE_GRID_DIM: usize = GRID_SIZE_HYPERPARAMETER * 4 + 2;
const FEATURE_GRID_SIZE: usize = FEATURE_GRID_DIM * FEATURE_GRID_DIM;

///
/// # This is used to compute the step size which maps
/// # from grid points to image points. (It is not the step size itself.)
/// # This is slightly bigger than the feature grid dimensions in order to
/// # make each region overlap slightly.
const FEATURE_STEP_DIVISOR: usize = GRID_SIZE_HYPERPARAMETER * 4 + 4;

/// # This is Equation 9 in the paper. It performs bilinear interpolation.
/// # The purpose of this is to better approximate the pixel information
/// # at a coordinate which is not an integer (and thus lies *between* pixels).
/// Function is aligned with Python impl
#[inline(always)]
fn interpolate_px_quad(summed_im: &[u64], im_w: usize, x: usize, y: usize, x_residue: f64, y_residue: f64) -> f64 {
    let px_1 = summed_im[y * im_w + x] as f64;
    let px_2 = summed_im[(y + 1) * im_w + x] as f64;
    let px_3 = summed_im[y * im_w + x + 1] as f64;
    let px_4 = summed_im[(y + 1) * im_w + x + 1] as f64;

    // NOTE: Must multiply the interpolation factors first *and then* the pixel
    // (due to rounding behavior)
    //println!("px x={x} y={y} im_w={im_w} {px_1} {px_2} {px_3} {px_4} | {px_avg}");

    ((1.0 - x_residue) * (1.0 - y_residue) * px_1)
        + ((1.0 - x_residue) * y_residue * px_2)
        + (x_residue * (1.0 - y_residue) * px_3)
        + (x_residue * y_residue * px_4)
}

/// Function is aligned with python impl
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn box_sum_for_radius(
    summed_im: &[u64],
    im_w: usize,
    im_h: usize,
    grid_step_h: f64,
    grid_step_v: f64,
    grid_point_x: f64,
    grid_point_y: f64,
    radius: f64,
    weight: f64,
) -> f64 {
    // Compute where the corners are. This is Equation 6.
    // NOTE: Parens required for rounding.
    let corner_a_x = (grid_point_x + (-radius * grid_step_h - 1.0)).clamp(0.0, im_w as f64 - 2.0);
    let corner_a_y = (grid_point_y + (-radius * grid_step_v - 1.0)).clamp(0.0, im_h as f64 - 2.0);
    let corner_d_x = (grid_point_x + radius * grid_step_h).clamp(0.0, im_w as f64 - 2.0);
    let corner_d_y = (grid_point_y + radius * grid_step_v).clamp(0.0, im_h as f64 - 2.0);

    // Get an integer pixel coordinate for the corners.
    // This is Equation 7.
    let corner_a_x_int = corner_a_x as usize;
    let corner_a_y_int = corner_a_y as usize;
    let corner_d_x_int = corner_d_x as usize;
    let corner_d_y_int = corner_d_y as usize;

    // Compute the fractional part, since we need it for interpolation.
    // This is Equation 8.
    let corner_a_x_residue = corner_a_x.fract();
    let corner_a_y_residue = corner_a_y.fract();
    let corner_d_x_residue = corner_d_x.fract();
    let corner_d_y_residue = corner_d_y.fract();

    //println!("corner r{radius} {corner_a_x} {corner_a_y} | {corner_d_x} {corner_d_y}");

    //println!("corner int r{radius} {corner_a_x_int} {corner_a_y_int} | {corner_d_x_int} {corner_d_y_int}");

    // Fetch the pixels in each corner
    let px_a = interpolate_px_quad(summed_im, im_w, corner_a_x_int, corner_a_y_int, corner_a_x_residue, corner_a_y_residue);

    let px_b = interpolate_px_quad(summed_im, im_w, corner_d_x_int, corner_a_y_int, corner_d_x_residue, corner_a_y_residue);
    let px_c = interpolate_px_quad(summed_im, im_w, corner_a_x_int, corner_d_y_int, corner_a_x_residue, corner_d_y_residue);

    let px_d = interpolate_px_quad(summed_im, im_w, corner_d_x_int, corner_d_y_int, corner_d_x_residue, corner_d_y_residue);

    //println!("box sum r{radius} {box_sum}");
    px_a * weight - px_b * weight - px_c * weight + px_d * weight
}

#[inline(always)]
fn compute_feature_grid(summed_im: &[u64], im_w: usize, im_h: usize) -> (FeatureGrid, f64, f64) {
    let grid_step_h = im_w as f64 / FEATURE_STEP_DIVISOR as f64;
    let grid_step_v = im_h as f64 / FEATURE_STEP_DIVISOR as f64;

    let mut feature_grid = [0.0; FEATURE_GRID_SIZE];
    (0..FEATURE_GRID_DIM).for_each(|feat_y| {
        (0..FEATURE_GRID_DIM).for_each(|feat_x| {
            //println!("-- grid {feat_x} {feat_y} --");
            let grid_point_x = (feat_x as f64 + 1.5) * grid_step_h;
            let grid_point_y = (feat_y as f64 + 1.5) * grid_step_v;

            //println!("grid point {grid_point_x} {grid_point_y}");

            let radius_box_0p2 = box_sum_for_radius(
                summed_im,
                im_w,
                im_h,
                grid_step_h,
                grid_step_v,
                grid_point_x,
                grid_point_y,
                0.2,
                WEIGHT_R1,
            );

            let radius_box_0p4 = box_sum_for_radius(
                summed_im,
                im_w,
                im_h,
                grid_step_h,
                grid_step_v,
                grid_point_x,
                grid_point_y,
                0.4,
                WEIGHT_R2,
            );

            let radius_box_0p8 = box_sum_for_radius(
                summed_im,
                im_w,
                im_h,
                grid_step_h,
                grid_step_v,
                grid_point_x,
                grid_point_y,
                0.8,
                WEIGHT_R3,
            );

            // Compute the final feature value. This is Equation 11.
            feature_grid[feat_y * FEATURE_GRID_DIM + feat_x] = radius_box_0p2 + radius_box_0p4 + radius_box_0p8
        })
    });

    (feature_grid, grid_step_h, grid_step_v)
}

// ----- (3.3) Gradient processing -----
#[inline(always)]
fn compute_gradient_grid(feature_grid: FeatureGrid) -> GradientGrid {
    let mut grad_out = [0.0; GRADIENT_GRID_SIZE];

    #[inline(always)]
    fn update_gradient_grid(gg: &mut GradientGrid, i: usize, v: f64) {
        //println!("gg[{i}] ({}) += {v}", gg[i]);
        gg[i] += v
    }

    (0..GRID_SIZE_HYPERPARAMETER).for_each(|feat_y_chunk| {
        (0..GRID_SIZE_HYPERPARAMETER).for_each(|feat_x_chunk| {
            (0..4).for_each(|feat_chunk_sub_y| {
                (0..4).for_each(|feat_chunk_sub_x| {
                    let feat_x = 1 + feat_x_chunk * 4 + feat_chunk_sub_x;
                    let feat_y = 1 + feat_y_chunk * 4 + feat_chunk_sub_y;

                    let feat_l = feature_grid[feat_y * FEATURE_GRID_DIM + feat_x - 1];
                    let feat_r = feature_grid[feat_y * FEATURE_GRID_DIM + feat_x + 1];
                    let feat_u = feature_grid[(feat_y - 1) * FEATURE_GRID_DIM + feat_x];
                    let feat_d = feature_grid[(feat_y + 1) * FEATURE_GRID_DIM + feat_x];

                    //println!("vals {feat_l} {feat_r} {feat_u} {feat_d}");

                    let grad_d_horiz = feat_l - feat_r;
                    let grad_d_vert = feat_u - feat_d;

                    let (grad_d_h_pos, grad_d_h_neg) = if grad_d_horiz <= 0.0 {
                        (0.0, -grad_d_horiz)
                    } else {
                        (grad_d_horiz, 0.0)
                    };

                    let (grad_d_v_pos, grad_d_v_neg) = if grad_d_vert <= 0.0 { (0.0, -grad_d_vert) } else { (grad_d_vert, 0.0) };

                    //println!("grad_d_h_pos={grad_d_h_pos} grad_d_h_neg={grad_d_h_neg} grad_d_v_pos={grad_d_v_pos} grad_d_v_neg={grad_d_v_neg}");

                    let grad_y_f = (feat_y as f64 - 2.5) * 0.25;
                    let grad_x_f = (feat_x as f64 - 2.5) * 0.25;
                    let grad_y = grad_y_f.floor();
                    let grad_x = grad_x_f.floor();
                    let grad_y_residue = grad_y_f - grad_y;
                    let grad_x_residue = grad_x_f - grad_x;

                    //println!("grad pos {grad_x} {grad_y} | {grad_x_residue} {grad_y_residue}");

                    let gg = &mut grad_out;

                    if grad_y >= 0.0 {
                        if grad_x >= 0.0 {
                            //println!("branch 1");
                            let i = ((grad_y as isize * GRID_SIZE_HYPERPARAMETER as isize + grad_x as isize) * 4) as usize;
                            let b = (1.0 - grad_x_residue) * (1.0 - grad_y_residue);
                            update_gradient_grid(gg, i, b * grad_d_h_pos);
                            update_gradient_grid(gg, i + 1, b * grad_d_h_neg);
                            update_gradient_grid(gg, i + 2, b * grad_d_v_pos);
                            update_gradient_grid(gg, i + 3, b * grad_d_v_neg);
                        }

                        if grad_x < 5.0 {
                            //println!("branch 2");
                            let i = ((grad_y as isize * GRID_SIZE_HYPERPARAMETER as isize + grad_x as isize + 1) * 4) as usize;
                            let b = grad_x_residue * (1.0 - grad_y_residue);
                            update_gradient_grid(gg, i, b * grad_d_h_pos);
                            update_gradient_grid(gg, i + 1, b * grad_d_h_neg);
                            update_gradient_grid(gg, i + 2, b * grad_d_v_pos);
                            update_gradient_grid(gg, i + 3, b * grad_d_v_neg);
                        }
                    }

                    if grad_y < 5.0 {
                        if grad_x >= 0.0 {
                            //println!("branch 3");
                            let i = (((grad_y as isize + 1) * GRID_SIZE_HYPERPARAMETER as isize + grad_x as isize) * 4) as usize;
                            let b = (1.0 - grad_x_residue) * grad_y_residue;
                            update_gradient_grid(gg, i, b * grad_d_h_pos);
                            update_gradient_grid(gg, i + 1, b * grad_d_h_neg);
                            update_gradient_grid(gg, i + 2, b * grad_d_v_pos);
                            update_gradient_grid(gg, i + 3, b * grad_d_v_neg);
                        }
                        if grad_x < 5.0 {
                            //println!("branch 4");
                            let i = (((grad_y as isize + 1) * GRID_SIZE_HYPERPARAMETER as isize + grad_x as isize + 1) * 4) as usize;
                            let b = grad_x_residue * grad_y_residue;
                            update_gradient_grid(gg, i, b * grad_d_h_pos);
                            update_gradient_grid(gg, i + 1, b * grad_d_h_neg);
                            update_gradient_grid(gg, i + 2, b * grad_d_v_pos);
                            update_gradient_grid(gg, i + 3, b * grad_d_v_neg);
                        }
                    }
                })
            })
        })
    });

    grad_out
}

// ----- (3.4) Hash normalization -----

const HASH_ITER_LIMIT: usize = 10;
const HASH_CLIP_CONST: f64 = 0.25;

/// The length of a PhotoDNA hash in bytes.
///
/// This constant defines the fixed size of all PhotoDNA hashes generated by this library.
pub const HASH_LEN: usize = GRADIENT_GRID_SIZE;

/// Fixed-size byte array type used for PhotoDNA hash storage.
///
/// This type represents the raw byte array that contains the PhotoDNA hash data.
/// The size is fixed at `HASH_LEN` bytes (currently 144 bytes).
///
/// # Examples
///
/// ```
/// use photo_dna::HashBuf;
///
/// // Create a zero-initialized hash buffer
/// let hash_buf: HashBuf = [0u8; 144];
/// ```
pub type HashBuf = [u8; HASH_LEN];

#[inline(always)]
fn empty_hashbuf() -> HashBuf {
    [0u8; HASH_LEN]
}

#[inline(always)]
fn normalize_hash(mut gradient_grid: GradientGrid, grid_step_h: f64, grid_step_v: f64) -> HashBuf {
    let scale_factor = grid_step_h * HASH_SCALE_CONST * grid_step_v * 3.0;
    gradient_grid.iter_mut().for_each(|g| *g /= scale_factor);

    for i in 0..HASH_ITER_LIMIT {
        let mut clip = false;

        // 1e-8 prevents div by 0 later
        let l2_norm = (gradient_grid.iter().map(|g| g.powi(2)).sum::<f64>() + 1e-8).sqrt();
        gradient_grid.iter_mut().for_each(|g| {
            // L2 normalization => equation (15)
            *g /= l2_norm;

            // Component clipping => equation (16)
            // Note: original implementation doesn't clip on the last iteration!
            if *g >= HASH_CLIP_CONST && i < HASH_ITER_LIMIT - 1 {
                *g = HASH_CLIP_CONST;
                clip = true;
            }
        });

        if !clip {
            break;
        }
    }

    hash_to_bytes(gradient_grid)
}

// This is Equation 17 in the paper
#[inline(always)]
fn hash_to_bytes(grid: GradientGrid) -> HashBuf {
    let mut hash = [0u8; GRADIENT_GRID_SIZE];
    grid.into_iter()
        .enumerate()
        .for_each(|(i, f)| hash[i] = (f * 256.0 / HASH_CLIP_CONST).clamp(0.0, 255.0) as u8);
    hash
}

/// A PhotoDNA hash representation.
///
/// This struct wraps a fixed-size byte array that represents the PhotoDNA hash
/// of an image. The hash can be used to compare images for similarity using
/// various distance metrics.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Hash(HashBuf);

#[cfg(feature = "serde")]
impl Serialize for Hash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bytes(self.as_bytes())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Hash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s: Vec<u8> = Deserialize::deserialize(deserializer)?;
        Self::from_bytes(s).map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

impl From<HashBuf> for Hash {
    fn from(value: HashBuf) -> Self {
        Self(value)
    }
}

/// Errors that can occur during PhotoDNA hash creation and manipulation.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid hash length error.
    ///
    /// This variant is returned when a hash operation is attempted with data
    /// that doesn't match the expected length requirements.
    #[error("invalid hash length: expected {expected}, got {actual}")]
    InvalidLength {
        /// The expected length in bytes that the operation required
        expected: usize,
        /// The actual length in bytes that was provided
        actual: usize,
    },

    /// Invalid hexadecimal string error.
    ///
    /// This variant is returned when attempting to create a hash from
    /// a hexadecimal string that is malformed or has incorrect length.
    #[error("invalid hex string")]
    InvalidHexString,

    /// Image processing error.
    ///
    /// This variant wraps errors that occur during image loading or processing.
    /// It typically originates from the underlying image crate.
    #[error("image: {0}")]
    Image(#[from] ImageError),

    /// Base64 decode error.
    ///
    /// This variant is returned when base64 decoding fails, typically due to
    /// invalid base64 input. Only available when the "base64" feature is enabled.
    #[cfg(feature = "base64")]
    #[error("base64 decode: {0}")]
    Base64Decode(#[from] DecodeError),
}

impl Hash {
    #[inline(always)]
    fn from_rgb_image(im: RgbImage) -> Result<Self, Error> {
        let (width, height) = (im.width(), im.height());
        let summed_pixels = preprocess_pixel_sum(im);
        let (feature_grid, grid_step_h, grid_step_v) = compute_feature_grid(&summed_pixels, width as usize, height as usize);

        let gradient_grid = compute_gradient_grid(feature_grid);
        Ok(normalize_hash(gradient_grid, grid_step_h, grid_step_v).into())
    }

    /// Creates a Hash from RGB pixel data.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the image in pixels
    /// * `height` - The height of the image in pixels
    /// * `pixels` - A vector of RGB pixel values, where each pixel is represented as [r, g, b]
    ///
    /// # Returns
    ///
    /// * `Ok(Hash)` - If the pixel data can be converted to a valid image
    /// * `Err(Error::InvalidLength)` - If the numbers pixels is different from `width * height`
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// // Create a small 2x2 red image
    /// let width = 2;
    /// let height = 2;
    /// let pixels = vec![[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]];
    /// let hash = Hash::from_rgb_pixels(width, height, pixels).unwrap();
    /// ```
    pub fn from_rgb_pixels(width: u32, height: u32, pixels: Vec<[u8; 3]>) -> Result<Self, Error> {
        let exp_len = (width as usize).saturating_mul(height as usize);

        if exp_len != pixels.len() {
            return Err(Error::InvalidLength {
                expected: exp_len,
                actual: pixels.len(),
            });
        }

        let rgb_data: Vec<u8> = pixels.into_iter().flatten().collect();
        if let Some(im) = RgbImage::from_vec(width, height, rgb_data) {
            return Self::from_rgb_image(im);
        }

        // we controlled pixels len against width and height so image should
        // not fail at being created (according to `RgbImage::from_vec` doc)
        unreachable!()
    }

    /// Creates a Hash from an image file at the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file
    ///
    /// # Returns
    ///
    /// * `Ok(Hash)` - If the image can be loaded and processed successfully
    /// * `Err(Error::Image)` - If there is an error loading the image file
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash = Hash::from_image_path("tests/random.png").unwrap();
    /// ```
    pub fn from_image_path<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let im = image::open(path)?;
        Self::from_rgb_image(im.into_rgb8())
    }

    /// Creates a Hash from a DynamicImage.
    ///
    /// # Arguments
    ///
    /// * `im` - A reference to a DynamicImage
    ///
    /// # Returns
    ///
    /// * `Ok(Hash)` - If the image can be processed successfully
    /// * `Err(Error::Image)` - If there is an error processing the image
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let img = image::open("tests/random.png").unwrap();
    /// let hash = Hash::from_image(&img).unwrap();
    /// ```
    pub fn from_image(im: &DynamicImage) -> Result<Self, Error> {
        Self::from_rgb_image(im.to_rgb8())
    }

    /// Creates a Hash from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `slice` - A byte slice that contains the hash data
    ///
    /// # Returns
    ///
    /// * `Ok(Hash)` - If the slice has the correct length (HASH_LEN)
    /// * `Err(Error::InvalidLength)` - If the slice length doesn't match HASH_LEN
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// // Create a hash from a valid byte array
    /// let hash_bytes = [0u8; 144];
    /// let hash = Hash::from_bytes(&hash_bytes).unwrap();
    /// ```
    #[inline]
    pub fn from_bytes<S: AsRef<[u8]>>(slice: S) -> Result<Self, Error> {
        let b = slice.as_ref();

        if b.len() != HASH_LEN {
            return Err(Error::InvalidLength {
                expected: HASH_LEN,
                actual: b.len(),
            });
        }

        let mut hash = empty_hashbuf();
        hash.copy_from_slice(b);
        Ok(hash.into())
    }

    /// Creates a Hash from a fixed-size byte array.
    ///
    /// # Arguments
    ///
    /// * `a` - A fixed-size byte array containing the hash data
    ///
    /// # Returns
    ///
    /// A Hash instance wrapping the provided byte array
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// // Create a hash from a zero-filled array
    /// let hash_array = [0u8; 144];
    /// let hash = Hash::from_array(hash_array);
    /// ```
    pub const fn from_array(a: HashBuf) -> Self {
        Self(a)
    }

    /// Returns the raw byte representation of the hash.
    ///
    /// # Returns
    ///
    /// A reference to the fixed-size byte array containing the hash data.
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash = Hash::from_image_path("tests/random.png").unwrap();
    /// let bytes = hash.as_bytes();
    /// println!("Hash length: {}", bytes.len());
    /// ```
    #[inline(always)]
    pub fn as_bytes(&self) -> &HashBuf {
        &self.0
    }

    /// Calculates the Euclidean distance between two hashes.
    ///
    /// This measures the straight-line distance between the two hash vectors
    /// in the multi-dimensional space. Lower values indicate more similar images.
    ///
    /// # Arguments
    ///
    /// * `other` - The other hash to compare with
    ///
    /// # Returns
    ///
    /// The Euclidean distance as an f64 value
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
    /// let hash2 = Hash::from_image_path("tests/random.png").unwrap();
    /// let distance = hash1.distance_euclid(&hash2);
    /// ```
    #[inline]
    pub fn distance_euclid(&self, other: &Self) -> f64 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(&x, &y)| (x as f64 - y as f64).powf(2.0))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculates the similarity between two hashes using Euclidean distance.
    ///
    /// Returns a normalized similarity score between 0.0 (completely different)
    /// and 1.0 (identical). This is calculated as 1 - (distance / max_possible_distance).
    ///
    /// # Arguments
    ///
    /// * `other` - The other hash to compare with
    ///
    /// # Returns
    ///
    /// A similarity score in the range [0.0, 1.0]
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
    /// let hash2 = Hash::from_image_path("tests/random.png").unwrap();
    /// let similarity = hash1.similarity_euclidian(&hash2);
    /// println!("Similarity: {:.2}%", similarity * 100.0);
    /// ```
    #[inline]
    pub fn similarity_euclidian(&self, other: &Self) -> f64 {
        let max_euclidian_distance = (HASH_LEN as f64 * 255.0 * 255.0).sqrt();
        1.0 - (self.distance_euclid(other) / max_euclidian_distance)
    }

    #[inline]
    /// Calculates the log2p distance between two hashes.
    ///
    /// This uses a logarithmic distance metric that gives more weight to
    /// smaller differences between hash values. This metric is often more
    /// effective for image similarity comparison than Euclidean distance.
    ///
    /// # Arguments
    ///
    /// * `other` - The other hash to compare with
    ///
    /// # Returns
    ///
    /// The log2p distance as an f64 value
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
    /// let hash2 = Hash::from_image_path("tests/random.png").unwrap();
    /// let distance = hash1.distance_log2p(&hash2);
    /// ```
    pub fn distance_log2p(&self, other: &Self) -> f64 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(&x, &y)| alog2p((x as i16 - y as i16).unsigned_abs() as u8))
            .sum::<f64>()
    }

    /// Calculates the similarity between two hashes using log2p distance.
    ///
    /// Returns a normalized similarity score between 0.0 (completely different)
    /// and 1.0 (identical) using the log2p distance metric. This is often more
    /// effective than Euclidean similarity for image comparison.
    ///
    /// # Arguments
    ///
    /// * `other` - The other hash to compare with
    ///
    /// # Returns
    ///
    /// A similarity score in the range [0.0, 1.0]
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
    /// let hash2 = Hash::from_image_path("tests/random.png").unwrap();
    /// let similarity = hash1.similarity_log2p(&hash2);
    /// println!("Similarity: {:.2}%", similarity * 100.0);
    /// ```
    #[inline]
    pub fn similarity_log2p(&self, other: &Self) -> f64 {
        const MAX_DISTANCE: f64 = HASH_LEN as f64 * alog2p(255);
        1.0 - (self.distance_log2p(other) / MAX_DISTANCE)
    }

    #[inline]
    /// Converts the hash to a hexadecimal string representation.
    ///
    /// # Returns
    ///
    /// A String containing the hexadecimal representation of the hash.
    /// Each byte is represented as two lowercase hexadecimal characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash = Hash::from_image_path("tests/random.png").unwrap();
    /// let hex_string = hash.to_hex_string();
    /// println!("Hash: {}", hex_string);
    /// ```
    pub fn to_hex_string(&self) -> String {
        let mut s = String::with_capacity(self.0.len() * 2);
        self.0.iter().for_each(|b| s.push_str(&format!("{:02x}", b)));
        s
    }

    /// Creates a Hash from a hexadecimal string representation.
    ///
    /// # Arguments
    ///
    /// * `s` - A hexadecimal string containing the hash data
    ///
    /// # Returns
    ///
    /// * `Ok(Hash)` - If the string is a valid hexadecimal representation with correct length
    /// * `Err(Error::InvalidHexString)` - If the string length is incorrect or contains invalid hex characters
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// // Create a hash from a hex string
    /// let hex_string = "deadbeef".repeat(36);
    /// let hash = Hash::from_hex_str(&hex_string).unwrap();
    /// ```
    #[inline]
    pub fn from_hex_str(s: &str) -> Result<Self, Error> {
        if s.len() != HASH_LEN * 2 {
            return Err(Error::InvalidHexString);
        }

        let mut hb = empty_hashbuf();

        macro_rules! hex_char_to_u8 {
            ($c: expr) => {{
                match $c as char {
                    '0'..='9' => $c - b'0',
                    'a'..='f' => $c - b'a' + 10,
                    'A'..='F' => $c - b'A' + 10,
                    _ => unreachable!(),
                }
            }};
        }

        for (i, c) in s.as_bytes().chunks_exact(2).enumerate() {
            if !c[0].is_ascii_hexdigit() || !c[1].is_ascii_hexdigit() {
                return Err(Error::InvalidHexString);
            } else {
                hb[i] = (hex_char_to_u8!(c[0]) << 4) + hex_char_to_u8!(c[1]);
            }
        }
        Ok(hb.into())
    }

    /// Converts the hash to a Base64 string representation.
    ///
    /// This method is only available when the "base64" feature is enabled.
    ///
    /// # Returns
    ///
    /// A Base64-encoded string representation of the hash.
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let hash = Hash::from_image_path("tests/random.png").unwrap();
    /// let base64_string = hash.to_base64_str();
    /// println!("Base64 hash: {}", base64_string);
    /// ```
    #[cfg(feature = "base64")]
    #[cfg_attr(docsrs, doc(cfg(feature = "base64")))]
    pub fn to_base64_str(&self) -> String {
        BASE64_STANDARD.encode(self.as_bytes())
    }

    /// Creates a Hash from a Base64 string representation.
    ///
    /// This method is only available when the "base64" feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `s` - A Base64-encoded string containing the hash data
    ///
    /// # Returns
    ///
    /// * `Ok(Hash)` - If the string is a valid Base64 representation
    /// * `Err(Error)` - If the Base64 decoding fails or the decoded data has incorrect length
    ///
    /// # Examples
    ///
    /// ```
    /// use photo_dna::Hash;
    ///
    /// let base64_string = "VTxBOEVeVVNBTENVYVFfYj9kVVNOOUZNSWc3XWt/XrFvUltFbFZ4U0xRSU8sa1M5QkNLOVZqgkqCO092VFhNVVxdfkA/YU5BQUI6PjJCVTxVSXBSWk06YWpqP3dNXV9gSRtJMExbXmZEUlltRFxGUHE7W4FVWmhyKScsSzBEVyk7Rlw+T2FJQG9JbkJfVFI+";
    /// let hash = Hash::from_base64_str(base64_string).unwrap();
    /// ```
    #[cfg(feature = "base64")]
    #[cfg_attr(docsrs, doc(cfg(feature = "base64")))]
    pub fn from_base64_str(s: &str) -> Result<Self, Error> {
        Self::from_bytes(BASE64_STANDARD.decode(s)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    const TEST_HASH: Hash = Hash::from_array([
        73, 71, 74, 32, 109, 76, 61, 78, 140, 80, 3, 105, 62, 51, 7, 76, 42, 38, 20, 34, 67, 17, 88, 23, 96, 59, 45, 61, 82, 202, 25, 115, 159, 37,
        16, 87, 62, 42, 40, 78, 73, 47, 13, 100, 89, 39, 24, 113, 174, 128, 42, 37, 98, 211, 71, 57, 156, 28, 47, 40, 106, 119, 50, 62, 140, 127, 46,
        83, 102, 25, 45, 47, 146, 88, 70, 6, 101, 177, 134, 17, 173, 107, 120, 17, 128, 154, 49, 40, 192, 137, 77, 15, 59, 82, 56, 30, 106, 46, 27,
        5, 129, 124, 64, 39, 90, 70, 101, 14, 50, 116, 126, 37, 146, 52, 127, 19, 23, 44, 85, 2, 124, 30, 11, 15, 98, 116, 15, 22, 102, 27, 15, 18,
        56, 105, 57, 28, 129, 61, 40, 65, 12, 20, 22, 9,
    ]);

    #[test]
    fn test_known_value() {
        assert_eq!(Hash::from_image_path("tests/image_2.jpg").unwrap(), TEST_HASH);
    }

    #[test]
    fn test_distance() {
        let h1 = Hash::from_image_path("tests/image_1.jpg").unwrap();
        let h2 = Hash::from_image_path("tests/random.png").unwrap();
        println!("Testing images with nothing in common");
        println!("similary euclidian={}", h1.similarity_euclidian(&h2));
        println!("similarity log2p={}", h1.similarity_log2p(&h2));

        println!("\nTesting similar images");
        let h3 = Hash::from_image_path("tests/image_2.jpg").unwrap();
        println!("similary euclidian={}", h1.similarity_euclidian(&h3));
        println!("similarity log2p={}", h1.similarity_log2p(&h3));
    }

    #[test]
    fn test_from_bytes_valid() {
        // Create a valid hash using Hash::from_image
        let valid_hash = Hash::from_image_path("tests/image_1.jpg").unwrap();
        let hash_bytes = valid_hash.as_bytes();

        // Test from_bytes with valid input
        let result = Hash::from_bytes(hash_bytes);
        assert!(result.is_ok());
        let hash_from_bytes = result.unwrap();

        // Verify the hash matches the original
        assert_eq!(hash_from_bytes.as_bytes(), hash_bytes);
    }

    #[test]
    fn test_from_bytes_invalid_length() {
        // Test with slice that's too short
        let short_slice = [0u8; 10];
        let result = Hash::from_bytes(short_slice);
        assert!(matches!(result, Err(Error::InvalidLength { expected, actual }) if expected == HASH_LEN && actual == 10));

        // Test with slice that's too long
        let long_slice = [0u8; 200];
        let result = Hash::from_bytes(long_slice);
        assert!(matches!(result, Err(Error::InvalidLength { expected, actual }) if expected == HASH_LEN && actual == 200));
    }

    #[test]
    fn test_from_bytes_different_types() {
        // Test with Vec<u8>
        let valid_hash = Hash::from_image_path("tests/image_1.jpg").unwrap();
        let hash_vec: Vec<u8> = valid_hash.as_bytes().to_vec();
        let result = Hash::from_bytes(hash_vec);
        assert!(result.is_ok());

        // Test with &[u8]
        let hash_slice: &[u8] = valid_hash.as_bytes();
        let result = Hash::from_bytes(hash_slice);
        assert!(result.is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_json_roundtrip() {
        use serde_json;

        // Create a hash from an image
        let original_hash = Hash::from_image_path("tests/image_1.jpg").unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&original_hash).unwrap();

        // Deserialize from JSON
        let deserialized_hash: Hash = serde_json::from_str(&json).unwrap();

        // Verify they are equal
        assert_eq!(original_hash, deserialized_hash);
        assert_eq!(original_hash.as_bytes(), deserialized_hash.as_bytes());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_json_invalid_length() {
        use serde_json;

        // Test with too short data
        let short_data = vec![0u8; 10];
        let json = serde_json::to_string(&short_data).unwrap();
        let result: Result<Hash, _> = serde_json::from_str(&json);
        assert!(result.is_err());

        // Test with too long data
        let long_data = vec![0u8; 200];
        let json = serde_json::to_string(&long_data).unwrap();
        let result: Result<Hash, _> = serde_json::from_str(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_impl() {
        let mut s = HashSet::new();
        assert!(s.insert(TEST_HASH));
        assert!(s.contains(&TEST_HASH));
        assert!(!s.insert(TEST_HASH));
    }
}
