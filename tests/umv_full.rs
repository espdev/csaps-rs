use ndarray::{array, Array1, Axis};
use approx::assert_abs_diff_eq;

use csaps::CubicSmoothingSpline;


const EPS: f64 = 1e-08;


#[test]
fn test_without_weights_auto_smooth_1() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![1.5, 3.5, 2.6, 1.2, 4.4];

    let s = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    assert_abs_diff_eq!(s.smooth().unwrap(), 0.8999999999999999);

    let coeffs_expected = array![[
        -0.41780962939499505,  0.39429046563192893,
         0.6284368070953434,  -0.6049176433322773,
         0.0,                 -1.253428888184985,
        -0.07055749128919839,  1.814752929996832,
         1.597869813113715,    0.3444409249287297,
        -0.979545454545454,    0.7646499841621794,
         1.7785397529299969,   2.958599936648717,
         2.44390243902439,     2.022236300285081,

     ]];

    assert_abs_diff_eq!(s.spline().unwrap().coeffs(), coeffs_expected, epsilon = EPS);

    let ys = s.evaluate(&x).unwrap();

    let ys_expected = array![
        1.7785397529299969, 2.958599936648717, 2.44390243902439, 2.022236300285081, 3.9967215711118156
    ];

    assert_abs_diff_eq!(ys, ys_expected, epsilon = EPS);
}


#[test]
fn test_without_weights_auto_smooth_2() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = array![1.5, 3.5, 2.6, 1.2, 4.4, 2.2, 1.6, 7.8, 9.1];

    let s = CubicSmoothingSpline::new(&x, &y)
        .make()
        .unwrap();

    let coeffs_expected = Array1::from(vec![
        -0.40104239278928844,     0.40119697598279225,
         0.4627405882617183,     -1.0601182231124329,
         0.8508522158245622,      0.6042037115821123,
        -1.312753456982271,       0.4549205812328071,
         0.0,                    -1.2031271783678652,
         0.0004637495805114,      1.3886855143656665,
        -1.7916691549716321,      0.7608874925020545,
         2.5734986272483913,     -1.3647617436984212,
         1.5988545517483752,      0.3957273733805099,
        -0.8069360554068441,      0.5822132085393337,
         0.17922956793336786,    -0.8515520945362101,
         2.482834025214236,       3.6915709087642057,
         1.7673615951928592,      2.965173754151946,
         2.5589709251473827,      2.2152392075827683,
         3.1260197073753355,      2.3644323361616335,
         2.87797144570959,        6.6215506411899465,
    ]).insert_axis(Axis(0));

    assert_abs_diff_eq!(s.spline().unwrap().coeffs(), coeffs_expected, epsilon = EPS);

    let ys = s.evaluate(&x).unwrap();

    let ys_expected = array![
       1.7673615951928592, 2.965173754151946,  2.5589709251473827,
       2.2152392075827683, 3.1260197073753355, 2.3644323361616335,
       2.87797144570959,   6.6215506411899465, 9.403280387488538,
    ];

    assert_abs_diff_eq!(ys, ys_expected, epsilon = EPS);
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

    assert_abs_diff_eq!(s.spline().unwrap().coeffs(), coeffs_expected, epsilon = EPS);

    let ys = s.evaluate(&x).unwrap();

    let ys_expected = array![
        1.7816259351620949, 2.3935461346633415, 2.247780548628429, 2.226284289276808, 3.553117206982544
    ];

    assert_abs_diff_eq!(ys, ys_expected, epsilon = EPS);
}
