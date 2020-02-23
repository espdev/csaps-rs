use std::fmt::Debug;
use std::result;

use num_traits::{Float, NumCast};


#[derive(Debug)]
pub struct CubicSmoothingSpline<'a, T>
    where T: Float + Debug
{
    x: &'a Vec<T>,
    y: &'a Vec<T>,

    weights: Option<&'a Vec<T>>,
    smooth: Option<T>,

    order: Option<u32>,
    pieces: Option<u32>,
    coeffs: Option<Vec<T>>,
}


pub type Result<T> = result::Result<T, String>;


impl<'a, T> CubicSmoothingSpline<'a, T>
    where T: Float + Debug
{

    pub fn new(x: &'a Vec<T>, y: &'a Vec<T>) -> CubicSmoothingSpline<'a, T> {
        CubicSmoothingSpline {
            x,
            y,
            weights: None,
            smooth: None,
            order: None,
            pieces: None,
            coeffs: None,
        }
    }

    pub fn with_weights(mut self, weights: &'a Vec<T>) -> CubicSmoothingSpline<'a, T> {
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

        let weights_default = vec![NumCast::from(1.0).unwrap(); self.x.len()];
        let weights = self.weights.unwrap_or(&weights_default);

        Ok(self)
    }

    pub fn evaluate(&self, xi: &'a Vec<T>) -> Result<Vec<T>> {
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

        Ok(Vec::new())
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

    pub fn coeffs(&self) -> Option<&Vec<T>> {
        match &self.coeffs {
            Some(coeffs) => Some(coeffs),
            None => None,
        }
    }

    fn validate_data(&self) -> Result<()> {
        if self.x.len() != self.y.len() {
            return Err(
                format!("The sizes of `x` ({}) and `y` ({}) are mismatched",
                        self.x.len(), self.y.len())
            )
        }

        if self.x.len() < 2 {
            return Err(
                "The size of data vectors must be greater or equal to 2".to_string()
            )
        }

        if self.weights.is_some() {
            let w = self.weights.unwrap();

            if w.len() != self.x.len() {
                return Err(
                    format!("The sizes of `weights` ({}) and `x` ({}) are mismatched",
                            w.len(), self.x.len())
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


#[cfg(test)]
mod tests {
    use crate::CubicSmoothingSpline;

    #[test]
    fn test_new() {
        let x= vec![1., 2., 3., 4.];
        let y= vec![1., 2., 3., 4.];

        let spline = CubicSmoothingSpline::new(&x, &y);

        assert!(spline.weights.is_none());
        assert!(spline.smooth.is_none());

        assert!(spline.order.is_none());
        assert!(spline.pieces.is_none());
        assert!(spline.coeffs.is_none());
    }

    #[test]
    fn test_options() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4.];
        let w = vec![1., 1., 1., 1.];
        let s = 0.5;

        let spline = CubicSmoothingSpline::new(&x, &y)
            .with_weights(&w)
            .with_smooth(s);

        assert!(spline.weights.is_some());
        assert!(spline.smooth.is_some());
    }

    #[test]
    #[should_panic(expected = "The sizes of `x` (4) and `y` (5) are mismatched")]
    fn test_data_size_mismatch_error() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4., 5.];

        let spline = CubicSmoothingSpline::new(&x, &y);
        spline.make().unwrap();
    }

    #[test]
    #[should_panic(expected = "The sizes of `weights` (5) and `x` (4) are mismatched")]
    fn test_weights_size_mismatch_error() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4.];
        let w = vec![1., 2., 3., 4., 5.];

        let spline = CubicSmoothingSpline::new(&x, &y);
        spline.with_weights(&w).make().unwrap();
    }

    #[test]
    #[should_panic(expected = "`smooth` value must be in range 0..1, given -0.5")]
    fn test_smooth_less_than_error() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4.];
        let s = -0.5;

        let spline = CubicSmoothingSpline::new(&x, &y);
        spline.with_smooth(s).make().unwrap();
    }

    #[test]
    #[should_panic(expected = "`smooth` value must be in range 0..1, given 1.5")]
    fn test_smooth_greater_than_error() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4.];
        let s = 1.5;

        let spline = CubicSmoothingSpline::new(&x, &y);
        spline.with_smooth(s).make().unwrap();
    }

    #[test]
    fn test_make() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4.];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .make().unwrap();

        assert!(spline.order().is_some());
        assert!(spline.pieces().is_some());
        assert!(spline.coeffs().is_some());
    }

    #[test]
    fn test_evaluate() {
        let x = vec![1., 2., 3., 4.];
        let y = vec![1., 2., 3., 4.];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .make().unwrap();

        let ys = spline.evaluate(&x).unwrap();

        assert_eq!(ys, y);
    }
}
