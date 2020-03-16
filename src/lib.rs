//! # Cubic spline approximation (smoothing)
//!
//! csaps is a crate for univariate, multivariate and n-d grid data interpolation and approximation
//! using cubic smoothing splines.
//!
//! The crate provides functionality for computing and evaluating cubic smoothing splines.
//! It does not contain any spline analysis functions. Therefore, the package can be useful in
//! practical engineering tasks for data approximation and smoothing.
//!
//! # Algorithm and Implementation
//!
//! The crate implements cubic smooting spline algorithm proposed by Carl de Boor in his book
//! ["A Practical Guide to Splines"](https://www.springer.com/gp/book/9780387953663) and inspired by
//! code from MATLAB [CSAPS](https://www.mathworks.com/help/curvefit/csaps.html) function and Fortran
//! routine SMOOTH from [PGS](http://pages.cs.wisc.edu/~deboor/pgs/) (originally written by Carl de Boor).
//!
//! The algorithm implementation based on [ndarray](https://docs.rs/ndarray) and [sprs](https://docs.rs/sprs) crates.
//!
//! # Features
//!
//! The crate provides the following features:
//!
//! - univariate data smoothing (X and Y are 1-d arrays/vectors)
//! - multivariate data smoothing (X is a 1-d array and Y is n-d array)
//! - n-dimensional grid data (a surface or volume for example) smoothing. Currently, the feature is not implemented yet
//! - weighted smoothing
//! - automatic smoothing (automatic computing the smoothing parameter)
//! - computing natural cubic spline interpolant when smoothing parameter is equal to one
//!
//! # Quick Examples
//!
//! Univariate data auto smoothing
//!
//! ```rust
//! use ndarray::{array, Array1};
//! use csaps::CubicSmoothingSpline;
//!
//! let x = array![1., 2., 3., 4.];
//! let y = array![0.5, 1.2, 3.4, 2.5];
//! let xi = Array1::linspace(1., 4., 10);
//!
//! let yi = CubicSmoothingSpline::new(&x, &y)
//!     .make().unwrap()
//!     .evaluate(&xi).unwrap();
//!
//! println!("xi: {}", xi);
//! println!("yi: {}", yi);
//! ```
//!
//! Weighted multivariate (3-d) data smoothing:
//!
//! ```rust
//! use ndarray::{array, Array1};
//! use csaps::CubicSmoothingSpline;
//!
//! let x = array![1., 2., 3., 4.];
//! let y = array![[0.5, 1.2, 3.4, 2.5],
//!                [1.5, 6.7, 7.1, 5.4],
//!                [2.3, 3.4, 5.6, 4.2]];
//! let w = array![1., 0.7, 0.5, 1.];
//! let xi = Array1::linspace(1., 4., 10);
//!
//! let yi = CubicSmoothingSpline::new(&x, &y)
//!     .with_weights(&w)
//!     .with_smooth(0.8)
//!     .make().unwrap()
//!     .evaluate(&xi).unwrap();
//!
//! println!("xi: {}", xi);
//! println!("yi: {}", yi);
//! ```
//!
//! # Input and Output Data Types
//!
//! The input data sites and data values should be array-like containers with floating point items
//! (`f32` or `f64`). It can be `&ndarray::Array` or `ndarray::ArrayView`, or `&Vec<_>`, or `&[_]`.
//!
//! The output evaluated data is always `ndarray::Array`.
//!

mod errors;
mod ndarrayext;
mod sprsext;
mod validate;
mod sspumv;

use std::result;

/// Provides result type for `make` and `evaluate` methods
pub type Result<T> = result::Result<T, errors::CsapsError>;

pub use errors::CsapsError;
pub use sspumv::{NdSpline, CubicSmoothingSpline};
