use ndarray::{
    NdFloat,
    ArrayView1,
    ArrayView,
    Dimension,
};

use almost::AlmostEqual;

use crate::{Result, CsapsError::InvalidInputData};
use crate::validate::{validate_data_sites, validate_smooth_value};

use super::GridCubicSmoothingSpline;


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    pub(super) fn make_validate(&self) -> Result<()> {
        validate_xy(&self.x, self.y.view())?;

        if let Some(weights) = self.weights.as_ref() {
            validate_weights(&self.x, weights)?;
        }

        if let Some(smooth) = self.smooth.as_ref() {
            validate_smooth(smooth)?;
        }

        Ok(())
    }
}


pub(super) fn validate_xy<T, D>(x: &[ArrayView1<'_, T>], y: ArrayView<'_, T, D>) -> Result<()>
    where
        T: NdFloat + AlmostEqual,
        D: Dimension
{
    if x.len() != y.ndim() {
        return Err(
            InvalidInputData(
                format!("The number of `x` data site vectors ({}) is not equal to `y` data dimensionality ({})",
                        x.len(), y.ndim())
            )
        )
    }

    for (ax, (&xi, &ys)) in x
        .iter()
        .zip(y.shape().iter())
        .enumerate()
    {
        let xi_len = xi.len();

        if xi_len < 2 {
            return Err(
                InvalidInputData(
                    format!("The size of `x` site vectors must be greater or equal to 2, axis {}", ax)
                )
            )
        }

        validate_data_sites(xi.view())?;

        if xi_len != ys {
            return Err(
                InvalidInputData(
                    format!("`x` data sites vector size ({}) is not equal to `y` data size ({}) for axis {}",
                            xi_len, ys, ax)
                )
            )
        }
    }

    Ok(())
}


pub(super) fn validate_weights<T>(x: &[ArrayView1<'_, T>], w: &[Option<ArrayView1<'_, T>>]) -> Result<()>
    where
        T: NdFloat + AlmostEqual
{
    let x_len = x.len();
    let w_len = w.len();

    if w_len != x_len {
        return Err(
            InvalidInputData(
                format!("The number of `weights` vectors ({}) is not equal to the number of `x` vectors ({})",
                        w_len, x_len)
            )
        )
    }

    for (ax, (xi, wi)) in x.iter().zip(w.iter()).enumerate() {
        if let Some(wi_view) = wi {
            let xi_len = xi.len();
            let wi_len = wi_view.len();

            if wi_len != xi_len {
                return Err(
                    InvalidInputData(
                        format!("`weights` vector size ({}) is not equal to `x` vector size ({}) for axis {}",
                                wi_len, xi_len, ax)
                    )
                )
            }
        }
    }

    Ok(())
}


pub(super) fn validate_smooth<T>(smooth: &[Option<T>]) -> Result<()>
    where
        T: NdFloat
{
    for (ax, s_opt) in smooth.iter().enumerate() {
        if let Some(s) = s_opt {
            match validate_smooth_value(*s) {
                Ok(res) => (),
                Err(err) => {
                    return Err(InvalidInputData(format!("{} for axis {}", err, ax)))
                }
            };
        }
    }

    Ok(())
}
