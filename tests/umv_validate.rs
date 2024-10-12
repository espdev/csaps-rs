use csaps::CubicSmoothingSpline;
use ndarray::{array, Array0, Axis};

#[test]
#[should_panic(expected = "Data site values must satisfy the condition: x1 < x2 < ... < xN")]
fn test_sites_invalid_order_1() {
    let x = array![1., 2., 2., 4.];
    let y = array![1., 2., 3., 4.];

    CubicSmoothingSpline::new(&x, &y).make().unwrap();
}

#[test]
#[should_panic(expected = "Data site values must satisfy the condition: x1 < x2 < ... < xN")]
fn test_sites_invalid_order_2() {
    let x = array![1., 2., 3., 1.];
    let y = array![1., 2., 3., 4.];

    CubicSmoothingSpline::new(&x, &y).make().unwrap();
}

#[test]
#[should_panic(expected = "`y` has zero dimensionality")]
fn test_zero_ndim_y_error() {
    let x = array![1., 2., 3., 4.];
    let y = Array0::<f64>::zeros(());

    CubicSmoothingSpline::new(&x, &y).make().unwrap();
}

#[test]
#[should_panic(expected = "The shape[0] (5) of `y` data is not equal to `x` size (4)")]
fn test_data_size_mismatch_error() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4., 5.];

    CubicSmoothingSpline::new(&x, &y).make().unwrap();
}

#[test]
#[should_panic(expected = "`axis` value (1) is out of bounds `y` dimensionality (1)")]
fn test_axis_out_of_bounds_error() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];

    CubicSmoothingSpline::new(&x, &y)
        .with_axis(Axis(1))
        .make()
        .unwrap();
}

#[test]
#[should_panic(expected = "`weights` size (5) is not equal to `x` size (4)")]
fn test_weights_size_mismatch_error() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let w = array![1., 2., 3., 4., 5.];

    CubicSmoothingSpline::new(&x, &y)
        .with_weights(&w)
        .make()
        .unwrap();
}

#[test]
#[should_panic(expected = "`smooth` value must be in range 0..1, given -0.5")]
fn test_smooth_less_than_error() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let s = -0.5;

    CubicSmoothingSpline::new(&x, &y)
        .with_smooth(s)
        .make()
        .unwrap();
}

#[test]
#[should_panic(expected = "`smooth` value must be in range 0..1, given 1.5")]
fn test_smooth_greater_than_error() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let s = 1.5;

    CubicSmoothingSpline::new(&x, &y)
        .with_smooth(s)
        .make()
        .unwrap();
}

#[test]
#[should_panic(expected = "The spline has not been computed, use `make` method before")]
fn test_evaluate_not_valid_error() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let xi = array![1., 2., 3., 4.];

    let spline = CubicSmoothingSpline::new(&x, &y);

    spline.evaluate(&xi).unwrap();
}
