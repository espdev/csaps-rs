//! # Cubic spline approximation (smoothing)
//!
//! csaps is a crate for univariate, multivariate and n-dimensional grid data interpolation and approximation
//! using cubic smoothing splines.
//!
//! The crate provides functionality for computing and evaluating cubic smoothing splines.
//! It does not contain any spline analysis functions. Therefore, the package can be useful in
//! practical engineering tasks for data approximation and smoothing.
//!
//! # Algorithm and Implementation
//!
//! The crate implements cubic smooting spline algorithm proposed by Carl de Boor in his book
//! ["A Practical Guide to Splines"](https://www.springer.com/gp/book/9780387953663) and has been inspired by
//! code from MATLAB [CSAPS](https://www.mathworks.com/help/curvefit/csaps.html) function and Fortran
//! routine SMOOTH from [PGS](http://pages.cs.wisc.edu/~deboor/pgs/) (originally written by Carl de Boor).
//!
//! The algorithm implementation is based on [ndarray](https://docs.rs/ndarray) and [sprs](https://docs.rs/sprs) crates.
//!
//! # Features
//!
//! The crate provides the following features:
//!
//! - univariate data smoothing (X and Y are 1-d arrays/vectors)
//! - multivariate data smoothing (X is a 1-d array and Y is a n-d array)
//! - n-dimensional grid data (a surface or volume for example) smoothing
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
//! 2-d grid (surface) data smoothing
//!
//! ```
//! use ndarray::array;
//! use csaps::GridCubicSmoothingSpline;
//!
//! let x0 = array![1.0, 2.0, 3.0, 4.0];
//! let x1 = array![1.0, 2.0, 3.0, 4.0];
//! let x = vec![x0.view(), x1.view()];
//!
//! let xi0 = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
//! let xi1 = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
//! let xi = vec![xi0.view(), xi1.view()];
//!
//! let y = array![
//!     [0.5, 1.2, 3.4, 2.5],
//!     [1.5, 2.2, 4.4, 3.5],
//!     [2.5, 3.2, 5.4, 4.5],
//!     [3.5, 4.2, 6.4, 5.5],
//! ];
//!
//! let yi = GridCubicSmoothingSpline::new(&x, &y)
//!     .make().unwrap()
//!     .evaluate(&xi).unwrap();
//!
//! println!("xi: {:?}", xi);
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
//! In n-dimensional grid data case the input `x` and `weights` data must be a slice of `ArrayView1`,
//! but not a slice of `AsArray` array-like because `ndarray::Array` does not implement `AsRef` trait
//! currently. In the future we might be able to support `AsArray` in n-dimensional grid data case.
//!
//! # Performance Issues
//!
//! Currently, the performance of computation of smoothing splines might be very low for a large data.
//!
//! The algorithm of sparse matrices mutliplication in sprs crate is not optimized for large diagonal
//! matrices which causes a poor performance of computation of smoothing splines.
//! See [issue](https://github.com/vbarrielle/sprs/issues/184) for details.
//!

mod errors;
mod traits;
mod ndarrayext;
mod sprsext;
mod validate;
mod util;
mod umv;
mod ndg;

use std::result;

/// Provides result type for `make` and `evaluate` methods
pub type Result<T> = result::Result<T, errors::CsapsError>;

pub use errors::CsapsError;
pub use traits::Real;
pub use umv::{NdSpline, CubicSmoothingSpline};
pub use ndg::{NdGridSpline, GridCubicSmoothingSpline};
