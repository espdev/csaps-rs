use ndarray::{
    NdFloat,
    Dimension,
    Array,
    Array1,
    ArrayView1,
};

use crate::{
    CubicSmoothingSpline,
    Result,
    arrayfuncs,
    validatedata,
};


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat, D: Dimension
{
    pub(crate) fn make_spline(&mut self) -> Result<()> {
        let weights_default = Array1::ones(self.x.raw_dim());
        let weights = self.weights
            .map(|v| v.reborrow()) // without it we will get an error: "[E0597] `weights_default` does not live long enough"
            .unwrap_or(weights_default.view());

        let pcount = self.x.len();
        let dx = arrayfuncs::diff(self.x, None);

        validatedata::validate_sites_increase(&dx)?;

        panic!("not implemented");
        Ok(())
    }

    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array<T, D> {
        panic!("not implemented");
    }
}
