use almost::AlmostEqual;
use ndarray::NdFloat;
use sprs::{CsMatBase, MulAcc};

// use num_traits::{Num, Float};
use num_traits::{Num, MulAdd, Float};

use std::ops::{Add, Mul, AddAssign};

/// Floating-point element types `f32` and `f64`.
///
/// Trait `Real` is only implemented for `f32` and `f64`, including the traits
/// needed for computing smoothing splines, manipulating n-d arrays and sparse matrices and also
/// checking almost equality.
///
/// This trait can only be implemented by `f32` and `f64`.
// pub trait Real<T>:
//     num_traits::Zero
//     + NdFloat
//     + PartialEq
//     + AlmostEqual
//     + Clone
//     + Default
//     + MulAcc
//     + for<'r> DivAssign<&'r T>
//     + Mul<T>
// {
// }
pub trait Real<T>:
    Num + NdFloat + Default +  AlmostEqual + Float
    + for<'r> std::ops::DivAssign<&'r T>
    + MulAdd<Output = T> 
{
}
pub trait RealRef<S, T>: Add<S, Output=T> + Mul<S, Output=T> {}


impl Real<f32> for f32 {}
impl Real<f64> for f64 {}

impl RealRef<&f32,f32> for &f32 {}
impl RealRef<&f64,f64> for &f64 {}

// fn test<T>(
//     a: &CsMatBase<T, usize, Vec<usize>, Vec<usize>, Vec<T>, usize>,
//     b: &CsMatBase<T, usize, Vec<usize>, Vec<usize>, Vec<T>, usize>,
// ) where
//     T: Real<T>,
//     // for<'r> &'r T: Add<&'r T, Output = T>,
//     for<'r> &'r T: RealRef<&'r T, T>
// {
//     let c = a + b;
// }

// fn test2(
//     a: &CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>, usize>,
//     b: &CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>, usize>,
// ) {
//     let c = a + b;
// }
