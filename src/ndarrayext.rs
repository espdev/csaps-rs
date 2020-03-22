use ndarray::{
    Dimension,
    IntoDimension,
    Axis,
    Slice,
    Ix1,
    Ix2,
    AsArray,
    Array,
    Array1,
    ArrayView,
    ArrayView2,
    s
};

use almost;
use itertools::Itertools;

use crate::{
    Real,
    Result,
    CsapsError::{ReshapeFrom2d, ReshapeTo2d},
    util::dim_from_vec
};


pub fn diff<'a, T: 'a, D, V>(data: V, axis: Option<Axis>) -> Array<T, D>
    where
        T: Real,
        D: Dimension,
        V: AsArray<'a, T, D>
{
    let data_view = data.into();
    let axis = axis.unwrap_or_else(|| Axis(data_view.ndim() - 1));

    let head = data_view.slice_axis(axis, Slice::from(..-1));
    let tail = data_view.slice_axis(axis, Slice::from(1..));

    &tail - &head
}


pub fn to_2d<'a, T: 'a, D, I>(data: I, axis: Axis) -> Result<ArrayView2<'a, T>>
    where
        D: Dimension,
        I: AsArray<'a, T, D>,
{
    let data_view = data.into();
    let ndim = data_view.ndim();

    // Firstly, we should permute ND array axes by given axis for getting
    // right NxM 2d array where N is the ndim and M is the data size
    let mut axes_tmp: Vec<usize> = (0..ndim).collect();
    axes_tmp.remove(axis.0);
    axes_tmp.push(axis.0);

    let axes: D = dim_from_vec(ndim, axes_tmp);

    let shape = data_view.shape().to_vec();
    let numel: usize = shape.iter().product();
    let axis_size = shape[axis.0];
    let new_shape = [numel / axis_size, axis_size];

    match data_view.permuted_axes(axes).into_shape(new_shape) {
        Ok(view_2d) => Ok(view_2d),
        Err(error) => Err(
            ReshapeTo2d {
                input_shape: shape,
                output_shape: new_shape.to_vec(),
                axis: axis.0,
                source: error,
            }
        )
    }
}


pub fn to_2d_simple<'a, T: 'a, D>(data: ArrayView<'a, T, D>) -> Result<ArrayView2<'a, T>>
    where
        D: Dimension
{
    let ndim = data.ndim();
    let shape = data.shape().to_vec();
    let new_shape = [shape[0..(ndim - 1)].iter().product(), shape[ndim - 1]];

    match data.into_shape(new_shape) {
        Ok(data_2d) => Ok(data_2d),
        Err(error) => Err(
            ReshapeTo2d {
                input_shape: shape,
                output_shape: new_shape.to_vec(),
                axis: ndim - 1,
                source: error,
            }
        )
    }
}


pub fn from_2d<'a, T: 'a, D, S, I>(data: I, shape: S, axis: Axis) -> Result<ArrayView<'a, T, S::Dim>>
    where
        D: Dimension,
        S: IntoDimension<Dim = D>,
        I: AsArray<'a, T, Ix2>,
{
    let shape = shape.into_dimension();
    let ndim = shape.ndim();

    let mut new_shape_vec = shape.slice().to_vec();
    new_shape_vec.remove(axis.0);
    new_shape_vec.push(shape[axis.0]);

    let new_shape: D = dim_from_vec(ndim, new_shape_vec.clone());
    let data_view = data.into();

    match data_view.into_shape(new_shape) {
        Ok(view_nd) => {
            let mut axes_tmp: Vec<usize> = (0..ndim).collect();
            let end_axis = axes_tmp.pop().unwrap();
            axes_tmp.insert(axis.0, end_axis);

            let axes: D = dim_from_vec(ndim, axes_tmp);
            Ok(view_nd.permuted_axes(axes))
        },
        Err(error) => Err(
            ReshapeFrom2d {
                input_shape: data_view.shape().to_vec(),
                output_shape: new_shape_vec,
                axis: axis.0,
                source: error,
            }
        )
    }
}


/// Returns the indices of the bins to which each value in input array belongs
///
/// This code works if `bins` is increasing
pub fn digitize<'a, T: 'a, A, B>(arr: A, bins: B) -> Array1<usize>
    where
        T: Real,
        A: AsArray<'a, T, Ix1>,
        B: AsArray<'a, T, Ix1>,
{
    let arr_view = arr.into();
    let bins_view = bins.into();

    let mut indices = Array1::zeros((arr_view.len(),));
    let mut kstart: usize = 0;

    for (i, &a) in arr_view.iter().enumerate()
        .sorted_by(|e1, e2| e1.1.partial_cmp(e2.1).unwrap()) {

        let mut k = kstart;

        for bins_win in bins_view.slice(s![kstart..]).windows(2) {
            let bl = bins_win[0];
            let br = bins_win[1];

            if (a > bl || almost::equal(a, bl)) && a < br {
                indices[i] = k;
                kstart = k;
                break;
            }

            k += 1;
        }
    }

    indices
}


#[cfg(test)]
mod tests {
    use std::f64;
    use ndarray::{array, Array1, Axis, Ix1, Ix2, Ix3};
    use crate::ndarrayext::*;

    #[test]
    fn test_diff_1d() {
        let a = array![1., 2., 3., 4., 5.];

        assert_eq!(diff(&a, None),
                   array![1., 1., 1., 1.]);

        assert_eq!(diff(&a, Some(Axis(0))),
                   array![1., 1., 1., 1.]);
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
    fn test_to_2d_simple_from_1d() {
        let a = array![1, 2, 3, 4];
        assert_eq!(to_2d_simple(a.view()).unwrap(), array![[1, 2, 3, 4]]);
    }

    #[test]
    fn test_to_2d_simple_from_2d() {
        let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];
        assert_eq!(to_2d_simple(a.view()).unwrap(), array![[1, 2, 3, 4], [5, 6, 7, 8]]);
    }

    #[test]
    fn test_to_2d_simple_from_3d() {
        let a = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
        assert_eq!(to_2d_simple(a.view()).unwrap(), array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
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

    #[test]
    fn test_digitize_1() {
        let xi = Array1::<f64>::linspace(1., 5., 9);
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![0, 0, 1, 1, 2, 2, 3, 3, 3])
    }

    #[test]
    fn test_digitize_2() {
        let xi = Array1::<f64>::linspace(0., 7., 15);
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., 5., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4])
    }

    #[test]
    fn test_digitize_3() {
        let xi = Array1::<f64>::linspace(1.5, 4.5, 7);
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., 5., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![0, 1, 1, 2, 2, 3, 3])
    }

    #[test]
    fn test_digitize_4() {
        let xi = Array1::<f64>::linspace(1.5, 4.5, 13);
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., 5., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    }

    #[test]
    fn test_digitize_5() {
        let xi = Array1::<f64>::linspace(2.5, 4.5, 17);
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., 5., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    }

    #[test]
    fn test_digitize_6() {
        let xi = Array1::<f64>::linspace(2.5, 8.5, 13);
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., 5., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4])
    }

    #[test]
    fn test_digitize_not_increased() {
        let xi = array![1., 2., 1., 3., 3., 2., 1., 4., 5., 5., 4., 4., 3., 3., 2., 1.];
        let edges = array![f64::NEG_INFINITY, 2., 3., 4., 5., f64::INFINITY];

        let indices = digitize(&xi, &edges);

        assert_eq!(indices, array![0, 1, 0, 2, 2, 1, 0, 3, 4, 4, 3, 3, 2, 2, 1, 0])
    }
}
