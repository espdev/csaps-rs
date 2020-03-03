mod make_spline;
mod evaluate_spline;
mod validate_data;

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


use crate::Result;


pub struct CubicSmoothingSpline<'a, T, D>
    where T: NdFloat + Default, D: Dimension
{
    x: ArrayView1<'a, T>,
    y: ArrayView<'a, T, D>,

    axis: Option<Axis>,

    weights: Option<ArrayView1<'a, T>>,
    smooth: Option<T>,

    ndim: Option<usize>,
    order: Option<usize>,
    pieces: Option<usize>,
    coeffs: Option<Array2<T>>,

    is_valid: bool,
}


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat + Default, D: Dimension
{
    pub fn new<V, Nd>(x: V, y: Nd) -> Self
        where V: AsArray<'a, T>,
              Nd: AsArray<'a, T, D>
    {
        CubicSmoothingSpline {
            x: x.into(),
            y: y.into(),
            axis: None,
            weights: None,
            smooth: None,
            ndim: None,
            order: None,
            pieces: None,
            coeffs: None,
            is_valid: false,
        }
    }

    pub fn with_axis(mut self, axis: Axis) -> Self {
        self.invalidate();
        self.axis = Some(axis);
        self
    }

    pub fn with_weights<V>(mut self, weights: V) -> Self
        where V: AsArray<'a, T>
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

    pub fn evaluate<V>(&self, xi: V) -> Result<Array<T, D>>
        where V: AsArray<'a, T>
    {
        let xi = xi.into();

        self.evaluate_validate_data(&xi)?;
        let ys = self.evaluate_spline(xi);
        Ok(ys)
    }

    pub fn ndim(&self) -> Option<usize> {
        self.ndim
    }

    pub fn smooth(&self) -> Option<T> {
        self.smooth
    }

    pub fn order(&self) -> Option<usize> {
        self.order
    }

    pub fn pieces(&self) -> Option<usize> {
        self.pieces
    }

    pub fn coeffs(&self) -> Option<ArrayView2<'_, T>> {
        match &self.coeffs {
            Some(coeffs) => Some(coeffs.view()),
            None => None,
        }
    }

    pub fn is_valid(&self) -> bool {
        return self.is_valid
    }

    fn invalidate(&mut self) {
        self.order = None;
        self.pieces = None;
        self.coeffs = None;
        self.is_valid = false;
    }
}
