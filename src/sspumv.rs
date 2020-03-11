use ndarray::{
    NdFloat,
    Dimension,
    Axis,
    AsArray,
    Array,
    Array2,
    ArrayView,
    ArrayView1,
    ArrayView2,
};

use almost::AlmostEqual;

use crate::Result;

mod validate_data;
mod make_spline;
mod evaluate_spline;


/// N-dimensional spline PP-form representation
#[derive(Debug)]
pub struct NdSpline<'a, T: NdFloat>
{
    ndim: usize,
    order: usize,
    pieces: usize,
    breaks: ArrayView1<'a, T>,
    coeffs: Array2<T>,
}


impl<'a, T> NdSpline<'a, T>
    where T: NdFloat + AlmostEqual
{
    pub fn new(order: usize, breaks: ArrayView1<'a, T>, coeffs: Array2<T>) -> NdSpline<'a, T> {
        let c_shape = coeffs.shape();
        let ndim = c_shape[0];
        let pieces = c_shape[1] / order;

        NdSpline {
            ndim,
            order,
            pieces,
            breaks,
            coeffs,
        }
    }

    pub fn ndim(&self) -> usize { self.ndim }

    pub fn order(&self) -> usize { self.order }

    pub fn pieces(&self) -> usize { self.pieces }

    pub fn breaks(&self) -> ArrayView1<'_, T> { self.breaks.view() }

    pub fn coeffs(&self) -> ArrayView2<'_, T> { self.coeffs.view() }

    pub fn evaluate<X>(&self, xi: X) -> Array2<T>
        where X: AsArray<'a, T>
    {
        self.evaluate_spline(xi.into())
    }
}


/// N-dimensional (univariate/multivariate) smoothing spline calculator/evaluator
pub struct CubicSmoothingSpline<'a, T, D>
    where T: NdFloat, D: Dimension
{
    x: ArrayView1<'a, T>,
    y: ArrayView<'a, T, D>,

    axis: Option<Axis>,

    weights: Option<ArrayView1<'a, T>>,
    smooth: Option<T>,

    spline: Option<NdSpline<'a, T>>
}


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat + Default + AlmostEqual, D: Dimension
{
    pub fn new<X, Y>(x: X, y: Y) -> Self
        where X: AsArray<'a, T>,
              Y: AsArray<'a, T, D>
    {
        CubicSmoothingSpline {
            x: x.into(),
            y: y.into(),
            axis: None,
            weights: None,
            smooth: None,
            spline: None,
        }
    }

    pub fn with_axis(mut self, axis: Axis) -> Self {
        self.invalidate();
        self.axis = Some(axis);
        self
    }

    pub fn with_weights<W>(mut self, weights: W) -> Self
        where W: AsArray<'a, T>
    {
        self.invalidate();
        self.weights = Some(weights.into());
        self
    }

    pub fn with_smooth(mut self, smooth: T) -> Self {
        self.invalidate();
        self.smooth = Some(smooth);
        self
    }

    pub fn make(mut self) -> Result<Self> {
        self.make_validate_data()?;
        self.make_spline()?;
        Ok(self)
    }

    pub fn evaluate<X>(&self, xi: X) -> Result<Array<T, D>>
        where X: AsArray<'a, T>
    {
        let xi = xi.into();
        self.evaluate_validate_data(xi)?;

        let yi = self.evaluate_spline(xi)?;
        Ok(yi)
    }

    pub fn smooth(&self) -> T {
        self.smooth.expect("The smoothing parameter yet has not been set or computed.")
    }

    pub fn spline(&self) -> &NdSpline<'a, T> {
        self.spline.as_ref().expect("The spline yet has not been computed.")
    }

    fn invalidate(&mut self) {
        self.spline = None;
    }
}
