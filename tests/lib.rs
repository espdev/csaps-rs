#[cfg(test)]
mod tests {
    use ndarray::array;
    use csaps::CubicSmoothingSpline;

    #[test]
    fn test_new() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.],
                                                [5., 6., 7., 8.]];

        let spline = CubicSmoothingSpline::new(&x, &y);

        assert!(spline.order().is_none());
        assert!(spline.pieces().is_none());
        assert!(spline.coeffs().is_none());
    }

    #[test]
    fn test_options() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.]];
        let w = array![1., 1., 1., 1.];
        let s = 0.5;

        let spline = CubicSmoothingSpline::new(&x, &y)
            .with_weights(&w)
            .with_smooth(s);

        assert!(spline.weights().is_some());
        assert!(spline.smooth().is_some());
    }

    #[test]
    #[should_panic(expected = "The shape[1] (5) of `y` data is not equal to `x` size (4)")]
    fn test_data_size_mismatch_error() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4., 5.]];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .make()
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "`weights` size (5) is not equal to `x` size (4)")]
    fn test_weights_size_mismatch_error() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.]];
        let w = array![1., 2., 3., 4., 5.];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .with_weights(&w)
            .make()
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "`smooth` value must be in range 0..1, given -0.5")]
    fn test_smooth_less_than_error() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.]];
        let s = -0.5;

        let spline = CubicSmoothingSpline::new(&x, &y)
            .with_smooth(s)
            .make()
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "`smooth` value must be in range 0..1, given 1.5")]
    fn test_smooth_greater_than_error() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.]];
        let s = 1.5;

        let spline = CubicSmoothingSpline::new(&x, &y)
            .with_smooth(s)
            .make()
            .unwrap();
    }

    #[test]
    fn test_make() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.]];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .make()
            .unwrap();

        assert!(spline.order().is_some());
        assert!(spline.pieces().is_some());
        assert!(spline.coeffs().is_some());
    }

    #[test]
    fn test_evaluate() {
        let x = array![1., 2., 3., 4.];
        let y = array![[1., 2., 3., 4.]];

        let spline = CubicSmoothingSpline::new(&x, &y)
            .make()
            .unwrap();

        let ys = spline.evaluate(&x).unwrap();

        assert_eq!(ys, y);
    }
}
