use ndarray::{
    NdFloat,
    Dimension,
    Axis,
    ArrayView1,
    Array1,
};

use crate::{
    CubicSmoothingSpline,
    Result,
    ndarrayext,
};


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat + Default, D: Dimension
{
    pub(crate) fn make_validate_data(&self) -> Result<()> {
        if self.y.ndim() == 0 {
            return Err(
                format!("`y` has zero dimensionality")
            )
        }

        let default_axis = Axis(self.y.ndim() - 1);
        let axis = self.axis.unwrap_or(default_axis);

        if axis > default_axis {
            return Err(
                format!("`axis` value ({}) is out of bounds `y` dimensionality ({})",
                        axis.0, self.y.ndim())
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

        if let Some(smooth) = self.smooth {
            if smooth < T::zero() || smooth > T::one() {
                return Err(
                    format!("`smooth` value must be in range 0..1, given {:?}", smooth)
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

        if self.spline.is_none() {
            return Err(
                "The spline has not been computed, use `make` method before".to_string()
            )
        }

        Ok(())
    }
}


pub(crate) fn validate_sites_increase<T>(dx: &Array1<T>) -> Result<()>
    where T: NdFloat
{
    let is_increase = dx.mapv(|v| v > T::zero());

    if !ndarrayext::all(&is_increase) {
        return Err(
            "Data site values must satisfy the condition: x1 < x2 < ... < xN".to_string()
        )
    }

    Ok(())
}
