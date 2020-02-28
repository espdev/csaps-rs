use ndarray::{array, Array, Array2, Dimension, Array1, NdFloat};
use csaps::CubicSmoothingSpline;


#[test]
fn test_new() {
    let x = array![1., 2., 3., 4.];

    let y1 = array![1., 2., 3., 4.];
    let y2 = array![[1., 2., 3., 4.], [5., 6., 7., 8.]];
    let y3 = array![[[1., 2., 3.], [5., 6., 7.]], [[1., 2., 3.], [5., 6., 7.]]];

    let spline1 = CubicSmoothingSpline::new(&x, &y1);
    assert_eq!(spline1.ndim(), y1.ndim());

    let spline2 = CubicSmoothingSpline::new(&x, &y2);
    assert_eq!(spline2.ndim(), y2.ndim());

    let spline3 = CubicSmoothingSpline::new(&x, &y3);
    assert_eq!(spline3.ndim(), y3.ndim());
}

fn test_driver_make_nd_npt<T, D>(x: Array1<T>, y: Array<T, D>,
                                 order: u32, pieces: u32, coeffs: Array2<T>)
    where T: NdFloat + Default, D: Dimension
{
    let spline = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    assert!(spline.is_valid());
    assert_eq!(spline.order().unwrap(), order);
    assert_eq!(spline.pieces().unwrap(), pieces);
    assert_eq!(spline.coeffs().unwrap(), coeffs);
}

fn test_driver_make_nd_2pt<T, D>(x: Array1<T>, y: Array<T, D>, coeffs: Array2<T>)
    where T: NdFloat + Default, D: Dimension
{
    test_driver_make_nd_npt(x, y, 2, 1, coeffs);
}

fn test_driver_make_nd_4pt<T, D>(x: Array1<T>, y: Array<T, D>, coeffs: Array2<T>)
    where T: NdFloat + Default, D: Dimension
{
    test_driver_make_nd_npt(x, y, 4, 3, coeffs);
}

#[test]
fn test_make_1d_2pt() {
    test_driver_make_nd_2pt(
        array![1., 2.],
        array![1., 2.],
        array![[1., 1.]]
    );
}

#[test]
fn test_make_2d_2pt() {
    test_driver_make_nd_2pt(
        array![1., 2.],
        array![[1., 2.], [3., 5.]],
        array![[1., 1.], [2., 3.]]
    );
}

#[test]
fn test_make_3d_2pt() {
    test_driver_make_nd_2pt(
        array![1., 2.],
        array![[[1., 2.], [3., 5.]], [[2., 4.], [4., 7.]]],
        array![[1., 1.], [2., 3.], [2., 2.], [3., 4.]]
    );
}

#[test]
fn test_make_1d_4pt() {
    test_driver_make_nd_4pt(
        array![1., 2., 3., 4.],
        array![1., 2., 3., 4.],
        array![[0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 2., 3.]]
    );
}

#[test]
fn test_make_2d_4pt() {
    test_driver_make_nd_4pt(
        array![1., 2., 3., 4.],
        array![[1., 2., 3., 4.], [1., 3., 5., 7.]],
        array![
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 2., 3.],
            [0., 0., 0., 0., 0., 0., 2., 2., 2., 1., 3., 5.]]
    );
}

#[test]
fn test_make_3d_4pt() {
    test_driver_make_nd_4pt(
        array![1., 2., 3., 4.],
        array![[[1., 2., 3., 4.], [1., 3., 5., 7.]], [[2., 4., 6., 8.], [3., 4., 5., 6.]]],
        array![
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 2., 3.],
            [0., 0., 0., 0., 0., 0., 2., 2., 2., 1., 3., 5.],
            [0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 4., 6.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 3., 4., 5.]]
    );
}

#[test]
fn test_evaluate() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];

    let spline = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    let ys = spline.evaluate(&x).unwrap();

    assert_eq!(ys, y);
}
