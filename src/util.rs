use ndarray::Dimension;
use itertools::Itertools;


/// Creates a dimension from the vector with the specified ndim
pub(crate) fn dim_from_vec<D>(ndim: usize, dimv: Vec<usize>) -> D
    where
        D: Dimension
{
    let mut dim = D::zeros(ndim);
    dim.as_array_view_mut().iter_mut().set_from(dimv);
    dim
}
