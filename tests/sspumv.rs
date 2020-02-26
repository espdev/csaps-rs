use ndarray::{array};
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

#[test]
fn test_make() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];

    let spline = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    assert!(spline.is_valid());
    assert!(spline.order().is_some());
    assert!(spline.pieces().is_some());
    assert!(spline.coeffs().is_some());
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

