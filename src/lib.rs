use std::fmt::Debug;
use std::result;

use num_traits::{Float, NumCast};
use ndarray::{Dimension, Array, Array1, Array2, ArrayView, ArrayView1, Axis};

mod sspumv_validate;
mod sspumv_impl;


#[derive(Debug)]
pub struct CubicSmoothingSpline<'a, T, D>
    where T: Float + Debug, D: Dimension
{
    x: ArrayView1<'a, T>,
    y: ArrayView<'a, T, D>,

    ndim: usize,
    axis: Option<Axis>,

    weights: Option<ArrayView1<'a, T>>,
    smooth: Option<T>,

    order: Option<u32>,
    pieces: Option<u32>,
    coeffs: Option<Array2<T>>,

    is_valid: bool,
}


pub type Result<T> = result::Result<T, String>;


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: Float + Debug, D: Dimension
{
    pub fn new(x: &'a Array1<T>, y: &'a Array<T, D>) -> Self {
        Self::from_view(x.view(), y.view())
    }

    pub fn from_view(x: ArrayView1<'a, T>, y: ArrayView<'a, T, D>) -> Self {
        let ndim = y.ndim();

        CubicSmoothingSpline {
            x,
            y,
            ndim,
            axis: None,
            weights: None,
            smooth: None,
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

    pub fn with_weights(mut self, weights: &'a Array1<T>) -> Self {
        self = self.with_weights_view(weights.view());
        self
    }

    pub fn with_weights_view(mut self, weights: ArrayView1<'a, T>) -> Self {
        self.invalidate();
        self.weights = Some(weights);
        self
    }

    pub fn with_smooth(mut self, smooth: T) -> Self {
        self.invalidate();
        self.smooth = Some(smooth);
        self
    }

    pub fn make(mut self) -> Result<Self> {
        self.make_validate_data()?;
        self.make_spline();
        Ok(self)
    }

    pub fn evaluate(&self, xi: &'a Array1<T>) -> Result<Array<T, D>> {
        self.evaluate_view(xi.view())
    }

    pub fn evaluate_view(&self, xi: ArrayView1<'a, T>) -> Result<Array<T, D>> {
        self.evaluate_validate_data(&xi)?;
        let ys = self.evaluate_spline(xi);
        Ok(ys)
    }

    pub fn ndim(&self) -> usize {
        self.ndim
    }

    pub fn smooth(&self) -> Option<T> {
        self.smooth
    }

    pub fn order(&self) -> Option<u32> {
        self.order
    }

    pub fn pieces(&self) -> Option<u32> {
        self.pieces
    }

    pub fn coeffs(&self) -> Option<&Array2<T>> {
        match &self.coeffs {
            Some(coeffs) => Some(coeffs),
            None => None,
        }
    }

    pub fn is_valid(&self) -> bool {
        return self.is_valid
    }
}
