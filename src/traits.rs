use ndarray::NdFloat;
use almost::AlmostEqual;


/// Floating-point element types `f32` and `f64`.
///
/// Trait `Real` is only implemented for `f32` and `f64`, including the traits
/// needed for computing smoothing splines, manipulating n-d arrays and sparse matrices and also
/// checking almost equality.
///
/// This trait can only be implemented by `f32` and `f64`.
pub trait Real: NdFloat + AlmostEqual + Default {}

impl Real for f32 {}
impl Real for f64 {}
