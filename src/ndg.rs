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
/// `NdGridSpline` represents n-dimensional splines for n-d grid data
///
/// Also `evaluate` method is implemented for `NdGridSpline` for evaluating the values
/// for given data sites.
///
#[derive(Debug)]
pub struct NdGridSpline<'a, T: NdFloat, D: Dimension>
{
    /// The grid dimensionality
    ndim: usize,

    /// The vector of the spline orders for each grid dimension
    order: Vec<usize>,

    /// The vector of the number of pieces of the spline for each grid dimension
    pieces: Vec<usize>,

    /// The breaks (data sites for each grid dimension) which have been used for computing spline
    breaks: &'a [ArrayView1<'a, T>],

    /// N-d array of the tensor-product univariate spline coefficients as
    /// representation of n-d grid spline coefficients
    coeffs: Array<T, D>,
}


///
pub struct NdGridCubicSmoothingSpline<'a, T, D>
    where T: NdFloat, D: Dimension
{
    /// X data sites (also breaks)
    x: &'a [ArrayView1<'a, T>],

    /// Y data values
    y: ArrayView<'a, T, D>,

    /// The optional data weights
    weights: Option<Vec<Option<ArrayView1<'a, T>>>>,

    /// The optional smoothing parameter
    smooth: Option<Vec<Option<T>>>,

    /// `NdSpline` struct with computed spline
    spline: Option<NdGridSpline<'a, T, D>>
}
