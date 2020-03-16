use std::convert::AsRef;

use ndarray::{
    NdFloat,
    Dimension,
    AsArray,
    Array,
    ArrayView,
    ArrayView1,
};

use almost::AlmostEqual;
use itertools::Itertools;

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
}


///
pub struct NdGridCubicSmoothingSpline<'a, T, D>
    where T: NdFloat, D: Dimension
{
    /// X data sites (also breaks)
    x: Vec<ArrayView1<'a, T>>,

    /// Y data values
    y: ArrayView<'a, T, D>,

    /// The optional data weights
    weights: Option<Vec<Option<ArrayView1<'a, T>>>>,

    /// The optional smoothing parameter
    smooth: Option<Vec<Option<T>>>,

    /// `NdSpline` struct with computed spline
    spline: Option<NdGridSpline<'a, T, D>>
}


impl<'a, T, D> NdGridCubicSmoothingSpline<'a, T, D>
    where T: NdFloat + AlmostEqual + Default,
          D: Dimension
{
    pub fn new<X, Y>(x: &'a [X], y: Y) -> Self
        where X: AsArray<'a, T> + AsRef<[T]>,
              Y: AsArray<'a, T, D>
    {
        NdGridCubicSmoothingSpline {
            x: x.iter().map_into().collect(),
            y: y.into(),
            weights: None,
            smooth: None,
            spline: None,
        }
    }
}
