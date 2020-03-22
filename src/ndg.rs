mod validate;
mod make;
mod evaluate;
mod util;

use ndarray::{
    Dimension,
    AsArray,
    Array,
    ArrayView,
    ArrayView1,
};

use crate::{Real, Result};


/// N-d grid spline PP-form representation
///
/// `NdGridSpline` represents n-dimensional splines for n-dimensional grid data. In n-d grid case
/// the spline is represented as tensor-product of univariate spline coefficients along every
/// diemension.
///
/// Also `evaluate` method is implemented for `NdGridSpline` for evaluating the values
/// for given data sites.
///
#[derive(Debug)]
pub struct NdGridSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    /// The grid dimensionality
    ndim: usize,

    /// The vector of the spline orders for each grid dimension
    order: Vec<usize>,

    /// The vector of the number of pieces of the spline for each grid dimension
    pieces: Vec<usize>,

    /// The breaks (data sites for each grid dimension) which have been used for computing spline
    breaks: Vec<ArrayView1<'a, T>>,

    /// N-d array of the tensor-product univariate spline coefficients as
    /// representation of n-d grid spline coefficients
    coeffs: Array<T, D>,
}


impl<'a, T, D> NdGridSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    /// Creates `NdGridSpline` struct from given `breaks` and `coeffs`
    ///
    /// # Arguments
    ///
    /// - `breaks` -- The vector of the breaks (data sites) which have been used for computing spline
    /// - `coeffs` -- The n-d array of tensor-product spline coefficients
    ///
    /// # Notes
    ///
    /// - `NdGridSpline` struct should not be created directly by a user in most cases.
    ///
    pub fn new(breaks: Vec<ArrayView1<'a, T>>, coeffs: Array<T, D>) -> Self {
        let ndim = breaks.len();
        let pieces: Vec<usize> = breaks.iter().map(|x| x.len() - 1).collect();
        let order: Vec<usize> = pieces.iter().zip(coeffs.shape().iter()).map(|(p, s)| s / p).collect();

        NdGridSpline {
            ndim,
            order,
            pieces,
            breaks,
            coeffs,
        }
    }

    /// Returns the n-d grid spline dimensionality
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the vector of the spline order for each dimension
    pub fn order(&self) -> &Vec<usize> { &self.order }

    /// Returns the vector of the number of pieces of the spline for each dimension
    pub fn pieces(&self) -> &Vec<usize> { &self.pieces }

    /// Returns the vector of views to the breaks for each dimension
    pub fn breaks(&self) -> &Vec<ArrayView1<'a, T>> { &self.breaks }

    /// Returns the view to the spline coefficients array
    pub fn coeffs(&self) -> ArrayView<'_, T, D> { self.coeffs.view() }

    /// Evaluates the spline on the given data sites
    pub fn evaluate(&self, xi: &'a [ArrayView1<'a, T>]) -> Array<T, D> {
        self.evaluate_spline(xi)
    }
}


/// N-dimensional grid cubic smoothing spline calculator/evaluator
///
/// The struct represents n-dimensional grid smoothing cubic spline and allows you to make
/// and evaluate the splines for given n-dimensional grid data.
///
/// `GridCubicSmoothingSpline` struct is parametrized by data type (`f64` or `f32`)
/// and data dimension.
///
/// The methods API of `GridCubicSmoothingSpline` is implemented as builder-like pattern or in other
/// words as chained API (also as `CubicSmoothingSpline` struct).
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use csaps::GridCubicSmoothingSpline;
///
/// let x0 = array![1.0, 2.0, 3.0, 4.0];
/// let x1 = array![1.0, 2.0, 3.0, 4.0];
/// let x = vec![x0.view(), x1.view()];
///
/// let y = array![
///     [0.5, 1.2, 3.4, 2.5],
///     [1.5, 2.2, 4.4, 3.5],
///     [2.5, 3.2, 5.4, 4.5],
///     [3.5, 4.2, 6.4, 5.5],
/// ];
///
/// let ys = GridCubicSmoothingSpline::new(&x, &y)
///     .make().unwrap()
///     .evaluate(&x).unwrap();
/// ```
///
pub struct GridCubicSmoothingSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    /// X data sites (also breaks)
    x: Vec<ArrayView1<'a, T>>,

    /// Y data values
    y: ArrayView<'a, T, D>,

    /// The optional data weights
    weights: Vec<Option<ArrayView1<'a, T>>>,

    /// The optional smoothing parameter
    smooth: Vec<Option<T>>,

    /// `NdGridSpline` struct with computed spline
    spline: Option<NdGridSpline<'a, T, D>>
}


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    /// Creates `NdGridCubicSmoothingSpline` struct from the given `X` data sites and `Y` data values
    ///
    /// # Arguments
    ///
    /// - `x` -- the slice of X-data sites 1-d array view for each dimension.
    ///   Each data sites must strictly increasing: `x1 < x2 < x3 < ... < xN`.
    /// - `y` -- The Y-data n-d grid values array-like. `ndim` can be from 1 to N.
    ///
    pub fn new<Y>(x: &[ArrayView1<'a, T>], y: Y) -> Self
        where
            Y: AsArray<'a, T, D>
    {
        let ndim = x.len();

        GridCubicSmoothingSpline {
            x: x.to_vec(),
            y: y.into(),
            weights: vec![None; ndim],
            smooth: vec![None; ndim],
            spline: None,
        }
    }

    /// Sets the weights data vectors for each dimension
    ///
    /// # Arguments
    ///
    /// - `weights` -- the slice of optional weights arrays (array-like) for each dimension
    ///
    /// # Notes
    ///
    /// `weights` vectors sizes must be equal to `x` data site sizes for each dimension.
    ///
    pub fn with_weights(mut self, weights: &[Option<ArrayView1<'a, T>>]) -> Self {
        self.invalidate();
        self.weights = weights.to_vec();
        self
    }

    /// Sets the smoothing parameters for each dimension
    ///
    /// # Arguments
    ///
    /// - `smooth` - the slice of optional smoothing parameters for each dimension
    ///
    /// # Notes
    ///
    /// The smoothing parameters should be in range `[0, 1]` or `None`,
    /// where bounds are:
    ///
    ///  - 0: The smoothing spline is the least-squares straight line fit to the data
    ///  - 1: The cubic spline interpolant with natural boundary condition
    ///
    /// If the smoothing parameter value is None, it will be computed automatically.
    ///
    pub fn with_smooth(mut self, smooth: &[Option<T>]) -> Self {
        self.invalidate();
        self.smooth = smooth.to_vec();
        self
    }

    /// Sets the smoothing parameter for all dimensions
    ///
    /// # Arguments
    ///
    /// - `smooth` - the smoothing parameter value that the same for all dimensions
    ///
    /// # Notes
    ///
    /// The smoothing parameter should be in range `[0, 1]`,
    /// where bounds are:
    ///
    ///  - 0: The smoothing spline is the least-squares straight line fit to the data
    ///  - 1: The cubic spline interpolant with natural boundary condition
    ///
    pub fn with_smooth_fill(mut self, smooth: T) -> Self {
        self.invalidate();
        self.smooth = vec![Some(smooth); self.x.len()];
        self
    }

    /// Makes (computes) the n-dimensional grid spline for given data and parameters
    ///
    /// # Errors
    ///
    /// - If the data or parameters are invalid
    ///
    pub fn make(mut self) -> Result<Self> {
        self.invalidate();
        self.make_validate()?;
        self.make_spline()?;

        Ok(self)
    }

    /// Evaluates the computed n-dimensional grid spline on the given data sites
    ///
    /// # Errors
    ///
    /// - If the `xi` data is invalid
    /// - If the spline yet has not been computed
    ///
    pub fn evaluate(&self, xi: &[ArrayView1<'a, T>]) -> Result<Array<T, D>> {
        self.evaluate_validate(&xi)?;
        let yi = self.evaluate_spline(&xi);

        Ok(yi)
    }

    /// Returns the ref to smoothing parameters vector or None
    pub fn smooth(&self) -> &Vec<Option<T>> {
        &self.smooth
    }

    /// Returns ref to `NdGridSpline` struct with data of computed spline or None
    pub fn spline(&self) -> Option<&NdGridSpline<'a, T, D>> {
        self.spline.as_ref()
    }

    /// Invalidate computed spline
    fn invalidate(&mut self) {
        self.spline = None;
    }
}
