use std::fmt::Debug;
use std::result;

use num_traits::{Float, NumCast};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};


#[derive(Debug)]
pub struct CubicSmoothingSpline<'a, T>
    where T: Float + Debug
{
    x: ArrayView1<'a, T>,
    y: ArrayView2<'a, T>,

    weights: Option<ArrayView1<'a, T>>,
    smooth: Option<T>,

    order: Option<u32>,
    pieces: Option<u32>,
    coeffs: Option<Array2<T>>,

    is_valid: bool,
}


pub type Result<T> = result::Result<T, String>;


impl<'a, T> CubicSmoothingSpline<'a, T>
    where T: Float + Debug
{
    pub fn new(x: &'a Array1<T>, y: &'a Array2<T>) -> CubicSmoothingSpline<'a, T> {
        Self::from_view(x.view(), y.view())
    }

    pub fn from_view(x: ArrayView1<'a, T>, y: ArrayView2<'a, T>) -> CubicSmoothingSpline<'a, T> {
        CubicSmoothingSpline {
            x,
            y,
            weights: None,
            smooth: None,
            order: None,
            pieces: None,
            coeffs: None,
            is_valid: false,
        }
    }

    pub fn with_weights(mut self, weights: &'a Array1<T>) -> CubicSmoothingSpline<'a, T> {
        self = self.with_weights_view(weights.view());
        self
    }

    pub fn with_weights_view(mut self, weights: ArrayView1<'a, T>) -> CubicSmoothingSpline<'a, T> {
        self.invalidate();
        self.weights = Some(weights);
        self
    }

    pub fn with_smooth(mut self, smooth: T) -> CubicSmoothingSpline<'a, T> {
        self.invalidate();
        self.smooth = Some(smooth);
        self
    }

    pub fn make(mut self) -> Result<CubicSmoothingSpline<'a, T>> {
        self.make_validate_data()?;

        let weights_default = Array1::ones(self.x.raw_dim());
        let weights = self.weights
            .map(|v| v.reborrow()) // without it we will get an error: "[E0597] `weights_default` does not live long enough"
            .unwrap_or(weights_default.view());

        Ok(self)
    }

    pub fn evaluate(&self, xi: &'a Array1<T>) -> Result<Array2<T>> {
        self.evaluate_view(xi.view())
    }

    pub fn evaluate_view(&self, xi: ArrayView1<'a, T>) -> Result<Array2<T>> {
        self.evaluate_validate_data(&xi)?;

        let arr = Array2::zeros(self.y.raw_dim());
        Ok(arr)
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

    fn make_validate_data(&self) -> Result<()> {
        let x_size = self.x.len();
        let y_size = self.y.len_of(Axis(1));

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

    fn evaluate_validate_data(&self, xi: &ArrayView1<'a, T>) -> Result<()> {
        if xi.len() < 2 {
            return Err(
                "The size of `xi` must be greater or equal to 2".to_string()
            )
        }

        if !self.is_valid {
            return Err(
                "The spline has not been computed, use `make` method before".to_string()
            )
        }

        Ok(())
    }

    fn invalidate(&mut self) {
        self.order = None;
        self.pieces = None;
        self.coeffs = None;
        self.is_valid = false;
    }
}
