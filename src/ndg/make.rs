use ndarray::{NdFloat, Dimension};
use almost::AlmostEqual;

use crate::{Result, CsapsError::ReshapeError, CubicSmoothingSpline};
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
        let mut coeffs = self.y.view();
        let mut coeffs_shape = coeffs.shape().to_vec();

        let mut smooth: Vec<Option<T>> = Vec::new();

        let mut permute_axes = D::zeros(ndim);
        permute_axes[0] = ndim_m1;
        for ax in 0..ndim_m1 {
            permute_axes[ax+1] = ax;
        }

        for ax in (0..ndim).rev() {
            let x = breaks[ax].view();

            if ndim > 2 {
                let coeffs_2d = {
                    let shape = coeffs.shape().to_vec();
                    let new_shape = [shape[0..ndim_m1].iter().product(), shape[ndim_m1]];

                    match coeffs.view().into_shape(new_shape) {
                        Ok(coeffs_2d) => coeffs_2d,
                        Err(err) => {
                            return Err(
                                ReshapeError(
                                    format!("Cannot reshape data array with shape {:?} to 2-d array with shape {:?}\n{}",
                                            shape, new_shape, err)
                                )
                            )
                        }
                    }
                };

            //     CubicSmoothingSpline::new(x, coeffs_2d.view())
            //         .make()?
            //         .spline().unwrap()
            //         .coeffs().to_owned()
            //
            // } else {
            //
            //     CubicSmoothingSpline::new(x, coeffs.view())
            //         .make()?
            //         .spline().unwrap()
            //         .coeffs().to_owned()
            }
        }

        self.smooth = Some(smooth);
        self.spline = Some(NdGridSpline::new(breaks, coeffs.to_owned()));

        Ok(())
    }
}
