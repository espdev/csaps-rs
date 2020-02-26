use num_traits::one;

use ndarray::{
    NdFloat,
    Dimension,
    Array,
    Array1,
    ArrayView1,
    Axis,
    s,
    stack,
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

        let dx = arrayfuncs::diff(self.x.view(), None);

        validatedata::validate_sites_increase(&dx)?;

        let axis = self.axis.unwrap_or(Axis(self.y.ndim() - 1));

        let y = arrayfuncs::to_2d(self.y.view(), axis)?;
        let dydx = arrayfuncs::diff(y.view(), Some(Axis(1))) / dx;

        let pcount = self.x.len();

        // The corner case for Nx2 data (2 data points)
        if pcount == 2 {
            let yi = y.slice(s![.., 0]).insert_axis(Axis(1));

            self.coeffs = Some(stack![Axis(1), dydx, yi]);
            self.smooth = Some(one());
            self.order = Some(2);
            self.pieces = Some(1);
            self.is_valid = true;

            return Ok(())
        }

        // General smoothing spline computing for NxM data (2 and more data points)

        panic!("not implemented");
        Ok(())
    }

    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array<T, D> {
        panic!("not implemented");
    }
}
