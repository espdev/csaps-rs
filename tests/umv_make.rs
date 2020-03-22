use ndarray::{array, Array, Array2, Dimension, Array1};
use csaps::{Real, CubicSmoothingSpline};


fn test_driver_make_nd_npt<T, D>(x: Array1<T>, y: Array<T, D>,
                                 order: usize, pieces: usize, coeffs: Array2<T>)
    where
        T: Real,
        D: Dimension
{
    let s = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    let spline = s.spline().unwrap();

    assert_eq!(spline.order(), order);
    assert_eq!(spline.pieces(), pieces);
    assert_eq!(spline.breaks(), x);
    assert_eq!(spline.coeffs(), coeffs);
}

fn test_driver_make_nd_2pt<T, D>(x: Array1<T>, y: Array<T, D>, coeffs: Array2<T>)
    where
        T: Real,
        D: Dimension
{
    test_driver_make_nd_npt(x, y, 2, 1, coeffs);
}

fn test_driver_make_nd_4pt<T, D>(x: Array1<T>, y: Array<T, D>, coeffs: Array2<T>)
    where
        T: Real,
        D: Dimension
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
fn test_make_2d_12x3_from_coeffs() {
    let x = array![1., 2., 3.];
    let y = array![
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 5., 9.],
        [2., 6., 10.],
        [3., 7., 11.],
    ];

    let spline = CubicSmoothingSpline::new(&x, &y)
        .make().unwrap();

    let coeffs_expected = array![
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 4., 4., 1., 5.],
        [0., 0., 0., 0., 4., 4., 2., 6.],
        [0., 0., 0., 0., 4., 4., 3., 7.],
    ];

    assert_eq!(spline.spline().unwrap().coeffs(), coeffs_expected)
}
