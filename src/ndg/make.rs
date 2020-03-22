use ndarray::Dimension;

use crate::{
    Real,
    Result,
    CubicSmoothingSpline,
    ndarrayext::to_2d_simple,
    util::dim_from_vec,
};

use super::{
    GridCubicSmoothingSpline,
    NdGridSpline,
    util::permute_axes
};


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    pub(super) fn make_spline(&mut self) -> Result<()> {
        let ndim = self.x.len();
        let ndim_m1 = ndim - 1;

        let breaks = self.x.to_vec();
        let mut coeffs = self.y.to_owned();
        let mut coeffs_shape = coeffs.shape().to_vec();

        let mut smooth: Vec<Option<T>> = vec![None; ndim];

        let permuted_axes: D = permute_axes(ndim);

        for ax in (0..ndim).rev() {
            let x = breaks[ax].view();
            let y = to_2d_simple(coeffs.view())?;

            let weights = self.weights[ax].map(|v| v.reborrow());
            let s = self.smooth[ax];

            let sp = CubicSmoothingSpline::new(x, y)
                .with_optional_weights(weights)
                .with_optional_smooth(s)
                .make()?;

            smooth[ax] = sp.smooth();

            coeffs = {
                let spline = sp.spline().unwrap();

                coeffs_shape[ndim_m1] = spline.pieces() * spline.order();
                let new_shape: D = dim_from_vec(ndim, coeffs_shape);

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
