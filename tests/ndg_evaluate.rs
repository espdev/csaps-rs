use approx::assert_abs_diff_eq;
use ndarray::array;

use csaps::GridCubicSmoothingSpline;

#[test]
fn test_make_vector_1() {
    let x0 = array![1., 2., 3., 4.];
    let x = vec![x0.view()];

    let y = array![1., 2., 3., 4.];

    let yi = GridCubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap()
        .evaluate(&x)
        .unwrap();

    assert_abs_diff_eq!(yi, y);
}

#[test]
fn test_make_vector_2() {
    let x0 = array![1., 2., 3., 4.];
    let x = vec![x0.view()];

    let xi0 = array![1., 1.5, 2., 2.5, 3., 3.5, 4.];
    let xi = vec![xi0.view()];

    let y = array![1., 2., 3., 4.];
    let yi_expected = array![1., 1.5, 2., 2.5, 3., 3.5, 4.];

    let yi = GridCubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap()
        .evaluate(&xi)
        .unwrap();

    assert_abs_diff_eq!(yi, yi_expected);
}

#[test]
fn test_make_surface_1() {
    let x0 = array![1., 2., 3.];
    let x1 = array![1., 2., 3., 4.];
    let x = vec![x0.view(), x1.view()];

    let y = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.],];

    let yi = GridCubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap()
        .evaluate(&x)
        .unwrap();

    assert_abs_diff_eq!(yi, y);
}

#[test]
fn test_make_surface_2() {
    let x0 = array![1., 2., 3.];
    let x1 = array![1., 2., 3., 4.];
    let x = vec![x0.view(), x1.view()];

    let xi0 = array![1., 1.5, 2., 2.5, 3.];
    let xi1 = array![1., 1.5, 2., 2.5, 3., 3.5, 4.];
    let xi = vec![xi0.view(), xi1.view()];

    let y = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.],];

    let yi_expected = array![
        [1., 1.5, 2., 2.5, 3., 3.5, 4.],
        [3., 3.5, 4., 4.5, 5., 5.5, 6.],
        [5., 5.5, 6., 6.5, 7., 7.5, 8.],
        [7., 7.5, 8., 8.5, 9., 9.5, 10.],
        [9., 9.5, 10., 10.5, 11., 11.5, 12.],
    ];

    let yi = GridCubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap()
        .evaluate(&xi)
        .unwrap();

    assert_abs_diff_eq!(yi, yi_expected);
}
