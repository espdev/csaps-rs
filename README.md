<p align="center">
  <a href="https://github.com/espdev/csaps-rs"><img src="https://user-images.githubusercontent.com/1299189/76691347-0a5ac780-665b-11ea-99fa-bf4a0aea04dc.png" alt="csaps" width="400" /></a><br>
</p>

<p align="center">
<a href="https://travis-ci.org/espdev/csaps-rs"><img src="https://travis-ci.org/espdev/csaps-rs.svg?branch=master" alt="Build status" /></a>
<a href="https://coveralls.io/github/espdev/csaps-rs?branch=master"><img src="https://coveralls.io/repos/github/espdev/csaps-rs/badge.svg?branch=master" alt="Coverage status" /></a>
</p>

<h4 align="center">
Cubic spline approximation (smoothing) algorithm written in Rust.
</h4>

**csaps** is a crate for univariate, multivariate and n-dimensional grid data approximation using cubic smoothing splines.
The package can be useful in practical engineering tasks for data approximation and smoothing.

## Usage

Multivariate data smoothing

```rust
use ndarray::{array, Array1};
use csaps::CubicSmoothingSpline;


fn main() {
    let x = array![1., 2., 3., 4.];
    let y = array![[1., 2., 3., 4.], 
                   [5., 6., 7., 8.]];
    let w = array![1., 0.7, 0.5, 1.];
    
    let spline = CubicSmoothingSpline::new(&x, &y)
        .with_weights(&w)
        .with_smoothing(0.8)
        .make()
        .unwrap();
    
    let xi = Array1::linspace(1., 4., 10);
    let yi = spline.evaluate(&xi).unwrap();
    
    println!("{}", xi);
    println!("{}", yi);
}
```

2-d grid (surface) data smoothing

```rust
use ndarray::array;
use csaps::GridCubicSmoothingSpline;


fn main() {
    let x0 = array![1.0, 2.0, 3.0, 4.0];
    let x1 = array![1.0, 2.0, 3.0, 4.0];
    let x = vec![x0.view(), x1.view()];
    
    let y = array![
        [0.5, 1.2, 3.4, 2.5],
        [1.5, 2.2, 4.4, 3.5],
        [2.5, 3.2, 5.4, 4.5],
        [3.5, 4.2, 6.4, 5.5],
    ];
    
    let yi = GridCubicSmoothingSpline::new(&x, &y)
     .with_smoothing_fill(0.5)
     .make().unwrap()
     .evaluate(&x).unwrap();
    
    println!("xi: {:?}", xi);
    println!("yi: {}", yi);
}
 ```

## Performance Issues

Currently, the performance of computation of smoothing splines might be very low for a large data.

The algorithm of sparse matrics mutliplication in sprs crate is not optimized for large diagonal 
matrices which causes a poor performance of computation of smoothing splines. 
See [issue](https://github.com/vbarrielle/sprs/issues/184) for details.


## Algorithms and implementations

The crate implementation is based on [ndaray](https://github.com/rust-ndarray/ndarray) and 
[sprs](https://github.com/vbarrielle/sprs) crates and has been inspired by Fortran routine SMOOTH from [PGS](http://pages.cs.wisc.edu/~deboor/pgs/) 
(originally written by Carl de Boor).

The implementation of the algotithm in other languages:
 
- [csaps in Python](https://github.com/espdev/csaps) Python NumPy/SciPy based implementation
- [csaps in C++](https://github.com/espdev/csaps-cpp) C++11 Eigen based implementation

## References

- C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978.

## License

[MIT](https://choosealicense.com/licenses/mit/)
