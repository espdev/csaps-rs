use ndarray::{
    Dimension,
    Axis,
    ArrayView1,
};

use crate::{
    Real,
    CubicSmoothingSpline,
    CsapsError::InvalidInputData,
    Result,
    validate::{validate_data_sites, validate_smooth_value},
};


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    pub(super) fn make_validate(&self) -> Result<()> {
        let x_size = self.x.len();

        if x_size < 2 {
            return Err(
                InvalidInputData(
                    "The size of data vectors must be greater or equal to 2".to_string()
                )
            )
        }

        validate_data_sites(self.x)?;

        if self.y.ndim() == 0 {
            return Err(
                InvalidInputData("`y` has zero dimensionality".to_string())
            )
        }

        let default_axis = Axis(self.y.ndim() - 1);
        let axis = self.axis.unwrap_or(default_axis);

        if axis > default_axis {
            return Err(
                InvalidInputData(
                    format!("`axis` value ({}) is out of bounds `y` dimensionality ({})",
                            axis.0, self.y.ndim())
                )
            )
        }

        let y_size = self.y.len_of(axis);

        if x_size != y_size {
            return Err(
                InvalidInputData(
                    format!("The shape[{}] ({}) of `y` data is not equal to `x` size ({})",
                            axis.0, y_size, x_size)
                )
            )
        }

        if let Some(weights) = self.weights {
            let w_size = weights.len();

            if w_size != x_size {
                return Err(
                    InvalidInputData(
                        format!("`weights` size ({}) is not equal to `x` size ({})", w_size, x_size)
                    )
                )
            }
        }

        if let Some(smooth) = self.smooth {
            validate_smooth_value(smooth)?;
        }

        Ok(())
    }

    pub(super) fn evaluate_validate(&self, xi: ArrayView1<'a, T>) -> Result<()> {
        if xi.is_empty() {
            return Err(
                InvalidInputData(
                    "The size of `xi` vector must be greater or equal to 1".to_string()
                )
            )
        }

        if self.spline.is_none() {
            return Err(
                InvalidInputData(
                    "The spline has not been computed, use `make` method before".to_string()
                )
            )
        }

        Ok(())
    }
}
