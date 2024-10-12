use csaps::CubicSmoothingSpline;
use ndarray::array;

#[test]
fn test_evaluate_1d() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let xi = array![1., 1.5, 2., 2.5, 3., 3.5, 4.];

    let spline = CubicSmoothingSpline::new(&x, &y).make().unwrap();

    let ys = spline.evaluate(&xi).unwrap();

    assert_eq!(ys, array![1., 1.5, 2., 2.5, 3., 3.5, 4.]);
}

#[test]
fn test_evaluate_2d_1() {
    let x = array![1., 2., 3., 4.];
    let y = array![[1., 2., 3., 4.], [3., 5., 7., 9.]];

    let spline = CubicSmoothingSpline::new(&x, &y).make().unwrap();

    let ys = spline.evaluate(&x).unwrap();

    assert_eq!(ys, y);
}

#[test]
fn test_evaluate_2d_2() {
    let x = array![1., 2., 3., 4.];
    let y = array![[1., 2., 3., 4.], [3., 5., 7., 9.]];
    let xi = array![1., 1.5, 2., 2.5, 3., 3.5, 4.];

    let spline = CubicSmoothingSpline::new(&x, &y).make().unwrap();

    let ys = spline.evaluate(&xi).unwrap();

    assert_eq!(
        ys,
        array![
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ]
    );
}

#[test]
fn test_evaluate_3d() {
    let x = array![1., 2., 3., 4.];
    let y = array![
        [[1., 2., 3., 4.], [2., 4., 6., 8.]],
        [[1., 3., 5., 7.], [3., 5., 7., 9.]]
    ];

    let spline = CubicSmoothingSpline::new(&x, &y).make().unwrap();

    let ys = spline.evaluate(&x).unwrap();

    assert_eq!(ys, y);
}
