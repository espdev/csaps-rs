use ndarray::{NdFloat, Dimension};
use almost::AlmostEqual;

use crate::Result;
use super::{GridCubicSmoothingSpline, NdGridSpline};


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    pub(super) fn make_spline(&mut self) -> Result<()> {
        unimplemented!();
    }
}
