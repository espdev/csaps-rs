use std::error::Error;

use num_traits::NumOps;

use ndarray::{
    ScalarOperand,
    Dimension,
    AsArray,
    Array,
    ArrayView2,
    Axis,
    Slice,
};

use crate::Result;


pub fn diff<'a, T: 'a, D, V>(data: V, axis: Option<Axis>) -> Array<T, D>
    where T: NumOps + ScalarOperand, D: Dimension, V: AsArray<'a, T, D>
{
    let data_view = data.into();
    let axis = axis.unwrap_or(Axis(data_view.ndim() - 1));

    let head = data_view.slice_axis(axis, Slice::from(..-1));
    let tail = data_view.slice_axis(axis, Slice::from(1..));

    &tail - &head
}


pub fn all<'a, D, V>(data: V) -> bool
    where D: Dimension, V: AsArray<'a, bool, D>
{
    for &v in data.into().iter() {
        if !v {
            return false
        }
    }

    true
}


pub fn to_2d<'a, T: 'a, D, I>(data: I, axis: Axis) -> Result<ArrayView2<'a, T>>
    where D: Dimension,
          I: AsArray<'a, T, D>,
{
    let data_view = data.into();
    let ndim = data_view.ndim();

    // Firstly, we should permute ND array axes by given axis for getting
    // right NxM 2d array where N is the ndim and M is the data size
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.remove(axis.0);
    axes.push(axis.0);

    // FIXME: it looks ugly, but...
    // I still do not know yet another way to get/create D-typed axes array
    // for passing to `permuted_axes` method to permute axes of D-dimensional array
    let mut axes_d = data_view.raw_dim();

    for (i, &ax) in axes.iter().enumerate() {
        axes_d[i] = ax
    }

    let shape = data_view.shape().to_vec();
    let numel: usize = shape.iter().product();
    let axis_size = shape[axis.0];
    let new_shape = [numel / axis_size, axis_size];

    match data_view.permuted_axes(axes_d).into_shape(new_shape) {
        Ok(view_2d) => Ok(view_2d),
        Err(error) => Err(
            format!("Cannot reshape {}-d array with shape {:?} by axis {} \
                    to 2-d array with shape {:?}. Error: {}",
                    ndim, shape, axis.0, new_shape, error.description()))
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{array, Axis};
    use crate::ndarrayext::*;

    #[test]
    fn test_diff_1d() {
        let a = array![1, 2, 3, 4, 5];

        assert_eq!(diff(&a, None),
                   array![1, 1, 1, 1]);

        assert_eq!(diff(&a, Some(Axis(0))),
                   array![1, 1, 1, 1]);
    }

    #[test]
    fn test_diff_2d() {
        let a = array![[1., 2., 3., 4.], [1., 2., 3., 4.]];

        assert_eq!(diff(&a, None),
                   array![[1., 1., 1.], [1., 1., 1.]]);

        assert_eq!(diff(&a, Some(Axis(0))),
                   array![[0., 0., 0., 0.]]);

        assert_eq!(diff(&a, Some(Axis(1))),
                   array![[1., 1., 1.], [1., 1., 1.]]);
    }

    #[test]
    fn test_diff_3d() {
        let a = array![[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]];

        assert_eq!(diff(&a, None),
                   array![[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]);

        assert_eq!(diff(&a, Some(Axis(0))),
                   array![[[0., 0., 0.], [0., 0., 0.]]]);

        assert_eq!(diff(&a, Some(Axis(1))),
                   array![[[0., 0., 0.]], [[0., 0., 0.]]]);

        assert_eq!(diff(&a, Some(Axis(2))),
                   array![[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]);
    }

    #[test]
    fn test_all_1d() {
        let a = array![true, true, true];
        assert!(all(&a));

        let a = array![true, false, true];
        assert!(!all(&a));
    }

    #[test]
    fn test_all_2d() {
        let a = array![[true, true, true], [true, true, true]];
        assert!(all(&a));

        let a = array![[true, true, true], [true, false, true]];
        assert!(!all(&a));
    }

    #[test]
    fn test_to_2d_from_1d() {
        let a = array![1, 2, 3, 4];

        assert_eq!(to_2d(&a, Axis(0)).unwrap(), array![[1, 2, 3, 4]]);
    }

    #[test]
    fn test_to_2d_from_2d() {
        let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];

        assert_eq!(to_2d(&a, Axis(0)).unwrap(), array![[1, 5], [2, 6], [3, 7], [4, 8]]);
        assert_eq!(to_2d(&a, Axis(1)).unwrap(), array![[1, 2, 3, 4], [5, 6, 7, 8]]);
    }

    #[test]
    fn test_to_2d_from_3d() {
        let a = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];

        // FIXME: incompatible memory layout
        // assert_eq!(to_2d(&a, Axis(0)).unwrap(), array![[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]]);
        // assert_eq!(to_2d(&a, Axis(1)).unwrap(), array![[1, 4], [2, 5], [3, 6], [7, 10], [8, 11], [9, 12]]);
        assert_eq!(to_2d(&a, Axis(2)).unwrap(), array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    }
}
