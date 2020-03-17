use ndarray::{NdFloat, Dimension, Array, ArrayView1};
use almost::AlmostEqual;

use crate::Result;
use super::{NdGridSpline, NdGridCubicSmoothingSpline};


impl<'a, T, D> NdGridSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual,
        D: Dimension
{
    /// Implements evaluating the spline on the given mesh of Xi-sites
    pub(super) fn evaluate_spline(&self, xi: &[ArrayView1<'a, T>]) -> Array<T, D> {
        unimplemented!();
    }
}


impl<'a, T, D> NdGridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    pub(super) fn evaluate_spline(&self, xi: &[ArrayView1<'a, T>]) -> Result<Array<T, D>> {
        unimplemented!();
    }
}
