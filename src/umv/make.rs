
use std::ops::{Add, Mul};

use ndarray::{prelude::*, concatenate, s};
use sprs::MulAcc;


use crate::{
    Real,
    Result,
    ndarrayext::{diff, to_2d},
    sprsext, RealRef
};

use super::{NdSpline, CubicSmoothingSpline};


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where
    T: Real<T>,
    for<'r> &'r T: RealRef<&'r T, T>,

    // T: MulAcc,
    // for<'r> &'r T: Add<&'r T, Output = T>,
    // for<'r> &'r T: Mul<&'r T, Output = T>,

    D: Dimension
{
    pub(super) fn make_spline(&mut self) -> Result<()> {
        // todo!();
        let one = T::one();
        let two = T::from::<f64>(2.0).unwrap();
        let three = T::from::<f64>(3.0).unwrap();
        let six = T::from::<f64>(6.0).unwrap();

        let breaks = self.x;

        let weights_default = Array1::ones(breaks.raw_dim());
        let weights = self.weights
            .map(|v| v.reborrow()) // without it we will get an error: "[E0597] `weights_default` does not live long enough"
            .unwrap_or_else(|| weights_default.view());

        let dx = diff(breaks.view(), None);

        let axis = self.axis.unwrap_or_else(|| Axis(self.y.ndim() - 1));
        self.axis = Some(axis);

        let y = to_2d(self.y.view(), axis)?;
        let dydx = diff(y.view(), Some(Axis(1))) / &dx;

        let pcount = breaks.len();

        // The corner case for Nx2 data (2 data points)
        if pcount == 2 {
            drop(dx);
            let yi = y.slice(s![.., 0 as i32]).insert_axis(Axis(1));
            let coeffs = concatenate![Axis(1), dydx, yi];

            self.smooth = Some(one);
            self.spline = Some(NdSpline::new(breaks, coeffs));

            return Ok(())
        }

        // General computing cubic smoothing spline for NxM data (3 and more data points)
        let ones = |n| Array1::<T>::ones((n, ));

        let qtwq = {
            let qt = {
                let odx = ones(pcount - 1) / &dx;
                let odx_head = odx.slice(s![..-1 as i32]).insert_axis(Axis(0)).into_owned();
                let odx_tail = odx.slice(s![1 as i32..]).insert_axis(Axis(0)).into_owned();
                drop(odx);
                let odx_body = -(&odx_tail + &odx_head);
                let diags_qt = concatenate![Axis(0), odx_head, odx_body, odx_tail];

                sprsext::diags(diags_qt, &[0, 1, 2], (pcount - 2, pcount))
            };


            let diags_sqrw = (ones(pcount) / weights.mapv(T::sqrt)).insert_axis(Axis(0));
            let sqrw = sprsext::diags(diags_sqrw, &[0], (pcount, pcount));
            let qtw = &qt * &sqrw;
            drop(sqrw);
            drop(qt);
            let qtw_t = qtw.transpose_view();

            &qtw * &qtw_t
        };

        let r = {
            let dx_head = dx.slice(s![..-1 as i32]).insert_axis(Axis(0)).into_owned();
            let dx_tail = dx.slice(s![1 as i32..]).insert_axis(Axis(0)).into_owned();
            let dx_body = (&dx_tail + &dx_head) * two;
            let diags_r = concatenate![Axis(0), dx_tail, dx_body, dx_head];

            sprsext::diags(diags_r, &[-1, 0, 1], (pcount - 2, pcount - 2))
        };

        let auto_smooth = || {
            let trace = |m| { sprsext::diagonal(m, 0).sum() };
            one / (one + trace(&r) / (six * trace(&qtwq)))
        };

        let smooth = self.smooth.unwrap_or_else(auto_smooth);
        let s1 = six * (one - smooth);

        // Solve linear system Ax = b for the 2nd derivatives
        let usol = {
            let a = {


                // cannot multiply `&CsMatBase<T, usize, Vec<usize>, Vec<usize>, Vec<T>>` by `T`
                // the trait `Mul<T>` is not implemented for `&CsMatBase<T, usize, Vec<usize>, Vec<usize>, Vec<T>>`
                
                let a1 = qtwq.map(|el| s1 * *el );
                let a2 = r.map(|el| *el * smooth);


                // let a1 = &qtwq * s1;
                // let a2 = &r * smooth;
                drop(qtwq);
                drop(r);


                &a1 + &a2 // was &a1 + &a2
            };

            let b = diff(&dydx, Some(Axis(1))).t().to_owned();
            drop(dydx);

            sprsext::solve(&a, &b)
        };

        // Compute and concatenatespline coefficients
        let coeffs = {
            let vpad = |arr: &Array2<T>| -> Array2<T> {
                let pad = Array2::<T>::zeros((1, arr.shape()[1]));
                concatenate(Axis(0), &[pad.view(), arr.view(), pad.view()]).unwrap()
            };

            let dx = dx.insert_axis(Axis(1));

            let yi = {
                let d1 = diff(&vpad(&usol), Some(Axis(0))) / &dx;
                let d2 = diff(&vpad(&d1), Some(Axis(0)));

                let diags_w = (ones(pcount) / weights).insert_axis(Axis(0));
                let w = sprsext::diags(diags_w, &[0], (pcount, pcount));
                let wd2 = &w * &d2;

                drop(d1);
                drop(d2);

                &y.t() - &wd2
            };

            let c3 = vpad(&(usol * smooth));
            let c3_head = c3.slice(s![..-1 as i32, ..]);
            let c3_tail = c3.slice(s![1 as i32.., ..]);

            let p1 = diff(&c3, Some(Axis(0))) / &dx;
            let p2 = &c3_head * three;
            let p3 = diff(&yi, Some(Axis(0))) / &dx - (&c3_head * two + c3_tail) * dx;
            let p4 = yi.slice(s![..-1 as i32, ..]); // was yi.view()

            drop(c3);

            concatenate(Axis(0), &[p1.view(), p2.view(), p3.view(), p4]).unwrap().t().to_owned()
        };

        self.smooth = Some(smooth);
        self.spline = Some(NdSpline::new(breaks, coeffs));

        Ok(())
    }
}
