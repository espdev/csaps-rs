use ndarray::{NdFloat, Dimension};
use almost::AlmostEqual;

use crate::{Result, CubicSmoothingSpline};
use crate::ndarrayext::to_2d_simple;

use super::{GridCubicSmoothingSpline, NdGridSpline, util::permute_axes};


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    pub(super) fn make_spline(&mut self) -> Result<()> {
        let ndim = self.x.len();
        let ndim_m1 = ndim - 1;

        let breaks = self.x.to_vec();
        let mut coeffs = self.y.to_owned();
        let mut coeffs_shape = coeffs.shape().to_vec();

        let mut smooth: Vec<Option<T>> = vec![None; ndim];

        let permuted_axes = permute_axes::<D>(ndim);

        for ax in (0..ndim).rev() {
            let x = breaks[ax].view();
            let y = to_2d_simple(coeffs.view())?;

            let weights = self.weights[ax].map(|v| v.reborrow());
            let s = self.smooth[ax];

            let sp = CubicSmoothingSpline::new(x, y)
                .with_optional_weights(weights)
                .with_optional_smoothing(s)
                .make()?;

            smooth[ax] = sp.smooth();

            coeffs = {
                let spline = sp.spline().unwrap();

                coeffs_shape[ndim_m1] = spline.pieces() * spline.order();
                let mut new_shape = D::zeros(ndim);
                for (ax, &sz) in coeffs_shape.iter().enumerate() {
                    new_shape[ax] = sz
                }

                spline.coeffs()
                    .into_shape(new_shape).unwrap()
                    .permuted_axes(permuted_axes.clone())
                    .to_owned()
            };

            coeffs_shape = coeffs.shape().to_vec();
        }

        self.smooth = smooth;
        self.spline = Some(NdGridSpline::new(breaks, coeffs));

        Ok(())
    }
}
