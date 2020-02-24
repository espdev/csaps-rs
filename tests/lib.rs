#[cfg(test)]
mod tests {
    use ndarray::{array, Array0, Axis};
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
    fn test_from_view() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.], [5., 6., 7., 8.]];

        let spline = CubicSmoothingSpline::from_view(x.view(), y.view());

        assert!(spline.order().is_none());
        assert!(spline.pieces().is_none());
        assert!(spline.coeffs().is_none());
    }

    #[test]
    #[should_panic(expected = "`y` has zero dimensionality")]
    fn test_zero_ndim_y_error() {
        let x = array![1., 2., 3., 4.];
        let y = Array0::<f64>::zeros(());

        CubicSmoothingSpline::new(&x, &y)
            .make()
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "The shape[0] (5) of `y` data is not equal to `x` size (4)")]
    fn test_data_size_mismatch_error() {
        let x = array![1., 2., 3., 4.];
        let y = array![1., 2., 3., 4., 5.];

        CubicSmoothingSpline::new(&x, &y)
            .make()
            .unwrap();
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

    #[test]
    #[should_panic(expected = "The size of `xi` must be greater or equal to 2")]
    fn test_evaluate_invalid_xi() {
        let x = array![1., 2., 3., 4.];
        let y = array![1., 2., 3., 4.];
        let xi = array![1.];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .make()
            .unwrap();

        spline.evaluate(&xi).unwrap();
    }
}
