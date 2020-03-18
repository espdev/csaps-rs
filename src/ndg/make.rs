use ndarray::{NdFloat, Dimension};
use almost::AlmostEqual;

use crate::{Result, CubicSmoothingSpline};
use crate::ndarrayext::to_2d_simple;

use super::{GridCubicSmoothingSpline, NdGridSpline};


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

        let mut permute_axes = D::zeros(ndim);
        permute_axes[0] = ndim_m1;
        for ax in 0..ndim_m1 {
            permute_axes[ax+1] = ax;
        }

        for ax in (0..ndim).rev() {
            let x = breaks[ax].view();
            let y = to_2d_simple(coeffs.view())?;

            let weights = self.weights[ax].map(|v| v.reborrow());
            let s = self.smooth[ax];

            println!("\nx shape: {:?}", x.shape());
            println!("y shape: {:?}", y.shape());

            println!("\nx: {:?}", x);
            println!("y: {:?}", y);

            let sp = CubicSmoothingSpline::new(x, y)
                .with_optional_weights(weights)
                .with_optional_smooth(s)
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
                    .permuted_axes(permute_axes.clone())
                    .to_owned()
            };

            coeffs_shape = coeffs.shape().to_vec();

            println!("\ncoeffs shape: {:?}", coeffs.shape());
            println!("coeffs: {:?}", coeffs);
        }

        self.smooth = smooth;
        self.spline = Some(NdGridSpline::new(breaks, coeffs));

        Ok(())
    }
}
