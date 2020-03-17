use ndarray::{NdFloat, Dimension};
use almost::AlmostEqual;

use crate::Result;
use super::{NdGridSpline, NdGridCubicSmoothingSpline};


impl<'a, T, D> NdGridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    pub(super) fn make_spline(&mut self) -> Result<()> {
        unimplemented!();
    }
}
