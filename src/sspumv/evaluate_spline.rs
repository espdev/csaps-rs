use ndarray::{NdFloat, Dimension, Array, Array2, ArrayView1};

use crate::{Result, ndarrayext};
use super::{CubicSmoothingSpline, NdSpline};


impl<'a, T: NdFloat> NdSpline<'a, T>
{
    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array2<T> {
        unimplemented!();
    }
}


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat, D: Dimension
{
    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Result<Array<T, D>> {
        let axis = self.axis.unwrap();
        let mut shape_t = self.y.shape().to_owned();
        shape_t[axis.0] = xi.len();

        let mut shape = D::zeros(self.y.ndim());

        for (i, &s) in shape_t.iter().enumerate() {
            shape[i] = s
        }

        let yi_2d = self.spline.as_ref().unwrap().evaluate(xi);
        let yi = ndarrayext::from_2d(&yi_2d, shape, axis)?.to_owned();

        Ok(yi)
    }
}
