use ndarray::{
    ArrayView,
    ArrayView1,
    Dimension,
};

use crate::{Real, Result, CsapsError::InvalidInputData};
use crate::validate::{validate_data_sites, validate_smooth_value};

use super::GridCubicSmoothingSpline;


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    pub(super) fn make_validate(&self) -> Result<()> {
        validate_xy(&self.x, self.y.view())?;
        validate_weights(&self.x, &self.weights)?;
        validate_smooth(&self.x, &self.smooth)?;

        Ok(())
    }

    pub(super) fn evaluate_validate(&self, xi: &[ArrayView1<'a, T>]) -> Result<()> {
        let x_len = self.x.len();
        let xi_len = xi.len();

        if xi_len != x_len {
            return Err(
                InvalidInputData(
                    format!("The number of `xi` vectors ({}) is not equal to the number of dimensions ({})",
                            xi_len, x_len)
                )
            )
        }

        for xi_ax in xi.iter() {
            if xi_ax.is_empty() {
                return Err(
                    InvalidInputData(
                        "The sizes of `xi` vectors must be greater or equal to 1".to_string()
                    )
                )
            }
        }

        Ok(())
    }
}


pub(super) fn validate_xy<T, D>(x: &[ArrayView1<'_, T>], y: ArrayView<'_, T, D>) -> Result<()>
    where
        T: Real,
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
        T: Real
{
    let x_len = x.len();
    let w_len = w.len();

    if w_len != x_len {
        return Err(
            InvalidInputData(
                format!("The number of `weights` vectors ({}) is not equal to the number of dimensions ({})",
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


pub(super) fn validate_smooth<T>(x: &[ArrayView1<'_, T>], smooth: &[Option<T>]) -> Result<()>
    where
        T: Real
{
    let x_len = x.len();
    let s_len = smooth.len();

    if s_len != x_len {
        return Err(
            InvalidInputData(
                format!("The number of `smooth` values ({}) is not equal to the number of dimensions ({})",
                        s_len, x_len)
            )
        )
    }

    for (ax, s_opt) in smooth.iter().enumerate() {
        if let Some(s) = s_opt {
            if let Err(err) = validate_smooth_value(*s) {
                return Err(InvalidInputData(format!("{} for axis {}", err, ax)))
            };
        }
    }

    Ok(())
}
