use num_traits::NumOps;

use ndarray::{
    ScalarOperand,
    Dimension,
    AsArray,
    Array,
    Array1,
    ArrayView,
    ArrayView2,
    Axis,
    Slice,
    Ix1,
    Ix2,
    IntoDimension
};

use crate::{CsapsError::ReshapeError, Result};


pub fn diff<'a, T: 'a, D, V>(data: V, axis: Option<Axis>) -> Array<T, D>
    where T: NumOps + ScalarOperand, D: Dimension, V: AsArray<'a, T, D>
{
    let data_view = data.into();
    let axis = axis.unwrap_or(Axis(data_view.ndim() - 1));

    let head = data_view.slice_axis(axis, Slice::from(..-1));
    let tail = data_view.slice_axis(axis, Slice::from(1..));

    &tail - &head
}


pub fn to_2d<'a, T: 'a, D, I>(data: I, axis: Axis) -> Result<ArrayView2<'a, T>>
    where D: Dimension,
          I: AsArray<'a, T, D>,
{
    let data_view = data.into();
    let ndim = data_view.ndim();

    // Firstly, we should permute ND array axes by given axis for getting
    // right NxM 2d array where N is the ndim and M is the data size
    let mut axes_t: Vec<usize> = (0..ndim).collect();
    axes_t.remove(axis.0);
    axes_t.push(axis.0);

    let mut axes = D::zeros(ndim);

    for (i, &ax) in axes_t.iter().enumerate() {
        axes[i] = ax
    }

    let shape = data_view.shape().to_vec();
    let numel: usize = shape.iter().product();
    let axis_size = shape[axis.0];
    let new_shape = [numel / axis_size, axis_size];

    match data_view.permuted_axes(axes).into_shape(new_shape) {
        Ok(view_2d) => Ok(view_2d),
        Err(error) => Err(
            ReshapeError(
                format!("Cannot reshape {}-d array with shape {:?} by axis {} \
                    to 2-d array with shape {:?}. Error: {}",
                    ndim, shape, axis.0, new_shape, error)
            ))
    }
}


pub fn from_2d<'a, T: 'a, D, S, I>(data: I, shape: S, axis: Axis) -> Result<ArrayView<'a, T, S::Dim>>
    where D: Dimension,
          S: IntoDimension<Dim = D>,
          I: AsArray<'a, T, Ix2>,
{
    let shape = shape.into_dimension();
    let ndim = shape.ndim();

    let shape_slice = shape.slice();

    let mut shape_t: Vec<usize> = shape_slice.to_owned();
    shape_t.remove(axis.0);
    shape_t.push(shape_slice[axis.0]);

    let mut new_shape = D::zeros(ndim);

    for (i, &s) in shape_t.iter().enumerate() {
        new_shape[i] = s;
    }

    let data_view = data.into();

    match data_view.into_shape(new_shape) {
        Ok(view_nd) => {
            let mut axes_t: Vec<usize> = (0..ndim).collect();
            let end_axis = axes_t.pop().unwrap();
            axes_t.insert(axis.0, end_axis);

            let mut axes = D::zeros(ndim);

            for (i, &ax) in axes_t.iter().enumerate() {
                axes[i] = ax
            }

            Ok(view_nd.permuted_axes(axes))
        },
        Err(error) => Err(
            ReshapeError(
                format!("Cannot reshape 2-d array with shape {:?} \
                    to {}-d array with shape {:?} by axis {}. Error: {}",
                        data_view.shape(), ndim, shape_t, axis.0, error)
            )
        )
    }
}


pub fn digitize<'a, T: 'a, I>(arr: I, bins: I) -> Array1<T>
    where I: AsArray<'a, T, Ix1>,
{
    unimplemented!();
}


#[cfg(test)]
mod tests {
    use ndarray::{array, Axis, Ix1, Ix2, Ix3};
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

    #[test]
    fn test_from_2d_to_3d() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        let e = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let s = e.shape();

        let r = from_2d(&a, s, Axis(2))
            .unwrap()
            .into_dimensionality::<Ix3>().unwrap();

        assert_eq!(r, e);
    }

    #[test]
    fn test_to_from_1d_axis0() {
        let a = array![1, 2, 3, 4, 5];
        let axis = Axis(0);

        let a_2d = to_2d(&a, axis).unwrap();
        let e = from_2d(&a_2d, a.shape(), axis)
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();

        assert_eq!(a_2d, array![[1, 2, 3, 4, 5]]);
        assert_eq!(a, e);
    }

    #[test]
    fn test_to_from_2d_axis0() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        let axis = Axis(0);

        let a_2d = to_2d(&a, axis).unwrap();
        let e = from_2d(&a_2d, a.shape(), axis)
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();

        assert_eq!(a_2d, array![[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]);
        assert_eq!(a, e);
    }

    #[test]
    fn test_to_from_2d_axis1() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        let axis = Axis(1);

        let a_2d = to_2d(&a, axis).unwrap();
        let e = from_2d(&a_2d, a.shape(), axis)
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();

        assert_eq!(a_2d, a);
        assert_eq!(a, e);
    }

    #[test]
    fn test_to_from_3d_axis2() {
        let a = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        let axis = Axis(2);

        let a_2d = to_2d(&a, axis).unwrap();
        let e = from_2d(&a_2d, a.shape(), axis)
            .unwrap()
            .into_dimensionality::<Ix3>()
            .unwrap();

        assert_eq!(a, e);
    }
}
