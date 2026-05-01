use ndarray::{concatenate, prelude::*, s};

use crate::{
    ndarrayext::{digitize, from_2d},
    util::dim_from_vec,
    Real, RealRef, Result,
};

use super::{CubicSmoothingSpline, NdSpline};

impl<'a, T> NdSpline<'a, T>
where
    T: Real<T>,
{
    /// Implements evaluating the spline on the given mesh of Xi-sites
    ///
    /// The internal method to avoid copying coeffs array
    pub(crate) fn evaluate_spline(
        order: usize,
        pieces: usize,
        breaks: ArrayView1<'_, T>,
        coeffs: ArrayView2<'_, T>,
        xi: ArrayView1<'a, T>,
    ) -> Array2<T> {
        let edges = {
            let mesh = breaks.slice(s![1 as i32..-1]);
            let one = Array1::<T>::ones((1,));
            let left_bound = &one * T::neg_infinity();
            let right_bound = &one * T::infinity();

            concatenate![Axis(0), left_bound, mesh, right_bound]
        };

        let mut indices = digitize(&xi, &edges);

        // Go to local coordinates
        let xi = {
            let indexed_breaks = indices.mapv(|i| breaks[i]);
            &xi - &indexed_breaks
        };

        // Apply nested multiplication

        // Returns NxM array of coeffs values for given 1xM indices array
        // where N is ndim and M is the size of xi
        let get_indexed_coeffs = |inds: &Array1<usize>| {
            let mut indexed_coeffs = Array2::<T>::zeros((coeffs.nrows(), inds.len()));

            for (col, &index) in inds.iter().enumerate() {
                indexed_coeffs
                    .column_mut(col)
                    .assign(&coeffs.slice(s![.., index]));
            }

            indexed_coeffs
        };

        // Vectorized computing the spline pieces (polynoms) on the given data sites
        let mut values = get_indexed_coeffs(&indices);

        for _ in 1..order {
            indices += pieces;
            values = values * &xi + get_indexed_coeffs(&indices);
        }

        values
    }
}

impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
where
    T: Real<T>,
    for<'r> &'r T: RealRef<&'r T, T>,

    D: Dimension,
{
    pub(super) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Result<Array<T, D>> {
        let axis = self.axis.unwrap();
        let mut shape_tmp = self.y.shape().to_owned();
        shape_tmp[axis.0] = xi.len();

        let shape: D = dim_from_vec(self.y.ndim(), shape_tmp);

        let yi_2d = self.spline.as_ref().unwrap().evaluate(xi);
        let yi = from_2d(&yi_2d, shape, axis)?.to_owned();

        Ok(yi)
    }
}
