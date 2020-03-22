# Changelog

## v0.3.0 (22.03.2020)

* Add `Real` pub trait. Trait `Real` is only implemented for `f32` and `f64`, 
  including the traits needed for computing smoothing splines, manipulating n-d arrays and 
  sparse matrices and also checking almost equality.
* Use `thiserror` crate for errors management
* Refactoring


## v0.2.1 (21.03.2020)

* Refactoring
* Fix typos in doc


## v0.2.0 (20.03.2020)

* API changes: rename smoothing parameter setters: `with_smoothing` -> `with_smooth`
* Slight fix docs


## v0.1.0 (20.03.2020)

* Initial public release
