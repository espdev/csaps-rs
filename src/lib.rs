use std::fmt::Debug;
use std::result;

use num_traits::{Float, NumCast};

use ndarray::{Array1, Array2, Axis, stack};


#[derive(Debug)]
pub struct CubicSmoothingSpline<'a, T>
    where T: Float + Debug
{
    x: &'a Array1<T>,
    y: &'a Array2<T>,

    axis: Axis,

    weights: Option<&'a Array1<T>>,
    smooth: Option<T>,

    order: Option<u32>,
    pieces: Option<u32>,
    coeffs: Option<Array2<T>>,
}


pub type Result<T> = result::Result<T, String>;


impl<'a, T> CubicSmoothingSpline<'a, T>
    where T: Float + Debug
{

    pub fn new(x: &'a Array1<T>, y: &'a Array2<T>) -> CubicSmoothingSpline<'a, T> {
        CubicSmoothingSpline {
            x,
            y,
            axis: Axis(1),
            weights: None,
            smooth: None,
            order: None,
            pieces: None,
            coeffs: None,
        }
    }

    pub fn with_weights(mut self, weights: &'a Array1<T>) -> CubicSmoothingSpline<'a, T> {
        self.clear();
        self.weights = Some(weights);
        self
    }

    pub fn with_smooth(mut self, smooth: T) -> CubicSmoothingSpline<'a, T> {
        self.clear();
        self.smooth = Some(smooth);
        self
    }

    pub fn make(mut self) -> Result<CubicSmoothingSpline<'a, T>> {
        self.validate_data()?;

        let weights_default = Array1::ones((self.x.len(),));
        let weights = self.weights.unwrap_or(&weights_default);

        Ok(self)
    }

    pub fn evaluate(&self, xi: &'a Array1<T>) -> Result<Array2<T>> {
        if xi.len() < 2 {
            return Err(
                "The size of `xi` must be greater or equal to 2".to_string()
            )
        }

        if self.order.is_none() {
            return Err(
                "The spline has not been computed, use `make` method before".to_string()
            )
        }

        let arr = Array2::zeros(self.y.raw_dim());
        Ok(arr)
    }

    pub fn weights(&self) -> Option<&'a Array1<T>> {
        self.weights
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

    fn validate_data(&self) -> Result<()> {
        let x_size = self.x.len();
        let y_size = self.y.len_of(self.axis);

        if x_size != y_size {
            return Err(
                format!("The shape[1] ({}) of `y` data is not equal to `x` size ({})", y_size, x_size)
            )
        }

        if x_size < 2 {
            return Err(
                "The size of data vectors must be greater or equal to 2".to_string()
            )
        }

        if self.weights.is_some() {
            let w = self.weights.unwrap();
            let w_size = w.len();

            if w_size != x_size {
                return Err(
                    format!("`weights` size ({}) is not equal to `x` size ({})", w_size, x_size)
                )
            }
        }

        if self.smooth.is_some() {
            let s = self.smooth.unwrap();
            let s_lower: T = NumCast::from(0.0).unwrap();
            let s_upper: T = NumCast::from(1.0).unwrap();

            if s < s_lower || s > s_upper {
                return Err(
                    format!("`smooth` value must be in range 0..1, given {:?}", s)
                )
            }
        }

        Ok(())
    }

    fn clear(&mut self) {
        self.order = None;
        self.pieces = None;
        self.coeffs = None;
    }
}
