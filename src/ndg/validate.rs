use ndarray::{
    NdFloat,
    ArrayView1,
    ArrayView,
    Dimension,
};

use almost::AlmostEqual;

use crate::{Result, CsapsError::InvalidInputData};
use crate::validate::validate_data_sites;


pub(super) fn validate_xy<'a, T, D>(x: &'a [ArrayView1<'a, T>], y: ArrayView<'a, T, D>) -> Result<()>
    where T: NdFloat + AlmostEqual, D: Dimension
{
    if x.len() != y.ndim() {
        return Err(
            InvalidInputData(
                format!("The number of `X` data sites ({}) is not equal to `Y` data dimensionality ({})",
                        x.len(), y.ndim())
            )
        )
    }

    for (ax, (&xi, &ys)) in x
        .iter()
        .zip(y.shape().iter())
        .enumerate()
    {
        validate_data_sites(xi.view())?;

        if xi.len() != ys {
            return Err(
                InvalidInputData(
                    format!("`X` data sites vector size ({}) is not equal to `Y` data size ({}) for axis {}",
                            xi.len(), ys, ax)
                )
            )
        }
    }

    Ok(())
}
