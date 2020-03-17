use ndarray::array;

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

    assert!(s.spline().is_some())
}
