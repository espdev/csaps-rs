use ndarray::{array, Array1};
use approx::assert_abs_diff_eq;

use csaps::GridCubicSmoothingSpline;


#[test]
fn test_make_surface() {
    let x0 = array![1., 2., 3.];
    let x1 = array![1., 2., 3., 4.];

    let x = vec![x0.view(), x1.view()];

    let y = array![
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
    ];

    let s = GridCubicSmoothingSpline::new(&x, &y)
        .make().unwrap();

    let coeffs_expected = array![
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 4., 4.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 4., 4.],
        [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 2., 3.],
        [0., 0., 0., 0., 0., 0., 1., 1., 1., 5., 6., 7.],
     ];

    let smooth: Array1<f64> = s.smooth().iter().map(|v| v.unwrap()).collect();

    assert_abs_diff_eq!(smooth, array![0.8999999999999999, 0.8999999999999999]);
    assert_abs_diff_eq!(s.spline().unwrap().coeffs(), coeffs_expected)
}
