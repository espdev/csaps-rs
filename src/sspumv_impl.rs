
use crate::{
    CubicSmoothingSpline,
    Float,
    Debug,
    Dimension,
    Array,
    Array1,
    ArrayView1,
};


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: Float + Debug, D: Dimension
{
    pub(crate) fn invalidate(&mut self) {
        self.order = None;
        self.pieces = None;
        self.coeffs = None;
        self.is_valid = false;
    }

    pub(crate) fn make_spline(&mut self) {
        let weights_default = Array1::ones(self.x.raw_dim());
        let weights = self.weights
            .map(|v| v.reborrow()) // without it we will get an error: "[E0597] `weights_default` does not live long enough"
            .unwrap_or(weights_default.view());

        panic!("not implemented");
    }

    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array<T, D> {
        panic!("not implemented");
    }
}
