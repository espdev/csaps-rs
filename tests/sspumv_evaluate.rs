use ndarray::array;
use csaps::CubicSmoothingSpline;


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
