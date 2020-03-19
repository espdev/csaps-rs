use ndarray::{NdFloat, Dimension, Array, ArrayView1};
use almost::AlmostEqual;

use crate::{NdSpline, ndarrayext::to_2d_simple};
use super::{NdGridSpline, GridCubicSmoothingSpline, util::permute_axes};


impl<'a, T, D> NdGridSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual,
        D: Dimension
{
    /// Implements evaluating the spline on the given mesh of Xi-sites
    pub(super) fn evaluate_spline(&self, xi: &[ArrayView1<'a, T>]) -> Array<T, D> {
        let mut coeffs = self.coeffs.to_owned();
        let mut coeffs_shape = coeffs.shape().to_vec();

        let ndim_m1 = self.ndim - 1;
        let permuted_axes = permute_axes::<D>(self.ndim);

        for ax in (0..self.ndim).rev() {
            let xi_ax = xi[ax];

            let coeffs_2d = {
                let coeffs_2d = to_2d_simple(coeffs.view()).unwrap();

                NdSpline::evaluate_spline(
                    self.order[ax],
                    self.pieces[ax],
                    self.breaks[ax],
                    coeffs_2d,
                    xi_ax,
                )
            };

            let mut shape = D::zeros(self.ndim);
            shape[ndim_m1] = xi_ax.len();
            for i in 0..ndim_m1 {
                shape[i] = coeffs_shape[i];
            }

            coeffs = coeffs_2d
                .into_shape(shape).unwrap()
                .permuted_axes(permuted_axes.clone())
                .to_owned();

            coeffs_shape = coeffs.shape().to_vec();
        }

        coeffs
    }
}


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    pub(super) fn evaluate_spline(&self, xi: &[ArrayView1<'a, T>]) -> Array<T, D> {
        self.spline.as_ref().unwrap().evaluate_spline(&xi)
    }
}
