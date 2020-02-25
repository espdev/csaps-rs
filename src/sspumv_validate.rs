
use crate::{
    CubicSmoothingSpline,
    Result,
    Float,
    NumCast,
    Debug,
    ArrayView1,
    Axis,
    Dimension,
};


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: Float + Debug, D: Dimension
{
    pub(crate) fn make_validate_data(&self) -> Result<()> {
        if self.ndim == 0 {
            return Err(
                format!("`y` has zero dimensionality")
            )
        }

        let default_axis = Axis(self.ndim - 1);
        let axis = self.axis.unwrap_or(default_axis);

        if axis > default_axis {
            return Err(
                format!("`axis` value ({}) is out of bounds `y` dimensionality ({})",
                        axis.0, self.ndim)
            )
        }

        let x_size = self.x.len();
        let y_size = self.y.len_of(axis);

        if x_size != y_size {
            return Err(
                format!("The shape[{}] ({}) of `y` data is not equal to `x` size ({})",
                        axis.0, y_size, x_size)
            )
        }

        if x_size < 2 {
            return Err(
                "The size of data vectors must be greater or equal to 2".to_string()
            )
        }

        if self.weights.is_some() {
            let w = self.weights.unwrap();
            let w_size = w.len();

            if w_size != x_size {
                return Err(
                    format!("`weights` size ({}) is not equal to `x` size ({})", w_size, x_size)
                )
            }
        }

        if self.smooth.is_some() {
            let s = self.smooth.unwrap();
            let s_lower: T = NumCast::from(0.0).unwrap();
            let s_upper: T = NumCast::from(1.0).unwrap();

            if s < s_lower || s > s_upper {
                return Err(
                    format!("`smooth` value must be in range 0..1, given {:?}", s)
                )
            }
        }

        Ok(())
    }

    pub(crate) fn evaluate_validate_data(&self, xi: &ArrayView1<'a, T>) -> Result<()> {
        if xi.len() < 2 {
            return Err(
                "The size of `xi` must be greater or equal to 2".to_string()
            )
        }

        if !self.is_valid {
            return Err(
                "The spline has not been computed, use `make` method before".to_string()
            )
        }

        Ok(())
    }
}
