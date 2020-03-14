<h1 align="center">
  <a href="https://github.com/espdev/csaps-rs"><img src="https://user-images.githubusercontent.com/1299189/76691347-0a5ac780-665b-11ea-99fa-bf4a0aea04dc.png" alt="csaps" width="400" /></a><br>
</h1>

:warning: :construction: experimental, work-in-progress :construction: :warning:

Cubic spline approximation (smoothing) algorithm written in Rust.

## Usage

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
        .with_smooth(0.8)
        .make()
        .unwrap();
    
    let xi = Array1::linspace(1., 4., 10);
    let yi = spline.evaluate(&xi).unwrap();
    
    println!("{}", xi);
    println!("{}", yi);
}
```

## References

- [csaps in Python](https://github.com/espdev/csaps)
- [csaps in C++](https://github.com/espdev/csaps-cpp)

## License

[MIT](https://choosealicense.com/licenses/mit/)
