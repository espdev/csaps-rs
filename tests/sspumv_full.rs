use ndarray::array;
use approx::assert_abs_diff_eq;

use csaps::CubicSmoothingSpline;


#[test]
fn test_without_weights_auto_smooth() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![1.5, 3.5, 2.6, 1.2, 4.4];

    let s = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    assert_abs_diff_eq!(s.smooth(), 0.8999999999999999);

    let coeffs_expected = array![[
        -0.41780962939499505,  0.39429046563192893,  0.6284368070953434,  -0.6049176433322773,
         0.0,                 -1.253428888184985,   -0.07055749128919839,  1.814752929996832,
         1.597869813113715,    0.3444409249287297,  -0.979545454545454,    0.7646499841621794,
         1.7785397529299969,   2.958599936648717,    2.44390243902439,     2.022236300285081,
     ]];

    assert_abs_diff_eq!(s.spline().coeffs(), coeffs_expected);

    let ys = s.evaluate(&x).unwrap();

    assert_abs_diff_eq!(ys, array![1.7785397529299969, 2.958599936648717, 2.44390243902439, 2.022236300285081, 3.9967215711118156]);
}


#[test]
fn test_with_weights_and_smooth() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![1.5, 3.5, 2.6, 1.2, 4.4];
    let w = array![1.0, 0.5, 0.7, 1.0, 0.6];

    let s = CubicSmoothingSpline::new(&x, &y)
        .with_weights(&w)
        .with_smooth(0.8)
        .make()
        .unwrap();

    let coeffs_expected = array![[
        -0.1877506234413966,  0.18106733167082298,  0.34543640897755623,  -0.33875311720698265,
         0.0,                -0.5632518703241898,  -0.020049875311720824,  1.016259351620948,
         0.7996708229426432,  0.23641895261845425, -0.34688279301745645,   0.6493266832917706,
         1.7816259351620949,  2.3935461346633415,   2.247780548628429,     2.226284289276808
     ]];

    assert_abs_diff_eq!(s.spline().coeffs(), coeffs_expected);

    let ys = s.evaluate(&x).unwrap();

    assert_abs_diff_eq!(ys, array![1.7816259351620949, 2.3935461346633415, 2.247780548628429, 2.226284289276808, 3.553117206982544]);
}
