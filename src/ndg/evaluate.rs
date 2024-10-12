use ndarray::{Array, ArrayView1, Dimension};

use crate::{ndarrayext::to_2d_simple, util::dim_from_vec, NdSpline, Real};

use super::{util::permute_axes, GridCubicSmoothingSpline, NdGridSpline};

impl<'a, T, D> NdGridSpline<'a, T, D>
where
    T: Real<T>,
    D: Dimension,
{
    /// Implements evaluating the spline on the given mesh of Xi-sites
    pub(super) fn evaluate_spline(&self, xi: &[ArrayView1<'a, T>]) -> Array<T, D> {
        let mut coeffs = self.coeffs.to_owned();
        let mut coeffs_shape = coeffs.shape().to_vec();

        let ndim_m1 = self.ndim - 1;
        let permuted_axes: D = permute_axes(self.ndim);

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

            coeffs = {
                coeffs_shape[ndim_m1] = xi_ax.len();
                let shape: D = dim_from_vec(self.ndim, coeffs_shape);

                coeffs_2d
                    .into_shape(shape)
                    .unwrap()
                    .permuted_axes(permuted_axes.clone())
                    .to_owned()
            };

            coeffs_shape = coeffs.shape().to_vec();
        }

        coeffs
    }
}

impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
where
    T: Real<T>,
    D: Dimension,
{
    pub(super) fn evaluate_spline(&self, xi: &[ArrayView1<'a, T>]) -> Array<T, D> {
        self.spline.as_ref().unwrap().evaluate_spline(&xi)
    }
}
