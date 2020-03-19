mod validate;
mod make;
mod evaluate;
mod util;

use ndarray::{
    NdFloat,
    Dimension,
    AsArray,
    Array,
    ArrayView,
    ArrayView1,
};

use almost::AlmostEqual;

use crate::Result;


/// N-d grid spline PP-form representation
///
/// `NdGridSpline` represents n-dimensional splines for n-d grid data. In n-d grid case
/// the spline is represented as tensor-product of univariate spline coefficients along every
/// diemnsion.
///
/// Also `evaluate` method is implemented for `NdGridSpline` for evaluating the values
/// for given data sites.
///
#[derive(Debug)]
pub struct NdGridSpline<'a, T, D>
    where
        T: NdFloat,
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
        T: NdFloat + AlmostEqual,
        D: Dimension
{
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

    /// Returns the spline dimensionality
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
/// The struct represents n-d grid smoothing cubic spline and allows you to make and evaluate the
/// splines for given n-d grid data.
///
/// `CubicSmoothingSpline` struct is parametrized by data type (`f64` or `f32`)
/// and data dimension.
///
pub struct GridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat,
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

    /// `NdSpline` struct with computed spline
    spline: Option<NdGridSpline<'a, T, D>>
}


impl<'a, T, D> GridCubicSmoothingSpline<'a, T, D>
    where
        T: NdFloat + AlmostEqual + Default,
        D: Dimension
{
    /// Creates `NdGridCubicSmoothingSpline` struct from the given `X` data sites and `Y` data values
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

    /// Sets the data weights
    ///
    /// `weights.len()` must be equal to `x.len()`
    ///
    pub fn with_weights(mut self, weights: &[Option<ArrayView1<'a, T>>]) -> Self {
        self.invalidate();
        self.weights = weights.to_vec();
        self
    }

    /// Sets the smoothing parameters for each axis
    ///
    /// The smoothing parameters should be in range `[0, 1]`,
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

    /// Makes (computes) the n-d grid spline for given data and parameters
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

    /// Evaluates the computed n-d grid spline on the given data sites
    ///
    /// # Errors
    ///
    /// - If the `xi` data is invalid
    /// - If the spline yet has not been computed
    ///
    pub fn evaluate(&self, xi: &[ArrayView1<'a, T>]) -> Result<Array<T, D>> {
        let xi = xi.to_vec();
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
