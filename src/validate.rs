use ndarray::ArrayView1;

use crate::{Real, Result, CsapsError::InvalidInputData};


pub(crate) fn validate_data_sites<T>(x: ArrayView1<T>) -> Result<()>
    where
        T: Real
{
    for w in x.windows(2) {
        let e1 = w[0];
        let e2 = w[1];
        if e2 < e1 || e2.almost_equals(e1) {
            return Err(
                InvalidInputData(
                    "Data site values must satisfy the condition: x1 < x2 < ... < xN".to_string()
                )
            )
        }
    }

    Ok(())
}


pub(crate) fn validate_smooth_value<T>(smooth: T) -> Result<()>
    where
        T: Real
{
    if smooth < T::zero() || smooth > T::one() {
        return Err(
            InvalidInputData(
                format!("`smooth` value must be in range 0..1, given {:?}", smooth)
            )
        )
    }

    Ok(())
}
