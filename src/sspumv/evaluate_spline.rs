use ndarray::{NdFloat, Dimension, Array, ArrayView1};
use crate::CubicSmoothingSpline;


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat + Default, D: Dimension
{
    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array<T, D> {
        unimplemented!();
    }
}
