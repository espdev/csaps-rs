use ndarray::NdFloat;
use almost::AlmostEqual;
use sprs::{MulAcc};

use std::ops::{Mul, DivAssign};


/// Floating-point element types `f32` and `f64`.
///
/// Trait `Real` is only implemented for `f32` and `f64`, including the traits
/// needed for computing smoothing splines, manipulating n-d arrays and sparse matrices and also
/// checking almost equality.
///
/// This trait can only be implemented by `f32` and `f64`.
pub trait Real<T>: NdFloat + AlmostEqual + Default + MulAcc + for<'r> DivAssign<&'r T> + Mul<T> {}

impl Real<f32> for f32 {}
impl Real<f64> for f64 {}

 