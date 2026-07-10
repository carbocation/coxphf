//! Literal Rust port of the native numerical routines in `coxphf.f90`.
//!
//! The first implementation intentionally retains the Fortran data layout and
//! control flow.  The Fortran entry points remain in the package as a numerical
//! reference while this backend is validated.

mod fit;
mod likelihood;
mod matrix;
mod native_data;
mod profile;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

#[no_mangle]
/// Run the Firth-Cox optimizer through the original three-array native ABI.
///
/// # Safety
///
/// `parms` must address 15 writable doubles. The dimensions encoded in it must
/// describe writable `cards` and `ioarray` buffers in the column-major layout
/// constructed by the R wrapper.
pub unsafe extern "C" fn rust_firthcox(cards: *mut f64, parms: *mut f64, ioarray: *mut f64) {
    if cards.is_null() || parms.is_null() || ioarray.is_null() {
        return;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let parms_slice = slice::from_raw_parts_mut(parms, 15);
        let n = parms_slice[0] as usize;
        let p = parms_slice[1] as usize;
        let ntde = parms_slice[13] as usize;
        let p_total = p + ntde;
        let cards_slice = slice::from_raw_parts(cards, n * (2 * p + 4 + 2 * ntde));
        let ioarray_slice = slice::from_raw_parts_mut(ioarray, (3 + p_total) * p_total);
        let data = native_data::NativeData::from_native_arrays(
            cards_slice,
            parms_slice,
            ioarray_slice,
            3 + p_total,
        );
        fit::fit(&data, parms_slice, ioarray_slice);
    }));

    if result.is_err() {
        *parms.add(7) = 98.0;
        *parms.add(9) = -98.0;
    }
}

#[no_mangle]
/// Compute profile likelihood intervals and tests through the native ABI.
///
/// # Safety
///
/// `parms` must address 15 writable doubles. The dimensions encoded in it must
/// describe writable `cards` and `ioarray` buffers in the column-major layout
/// constructed by the R wrapper.
pub unsafe extern "C" fn rust_plcomp(cards: *mut f64, parms: *mut f64, ioarray: *mut f64) {
    if cards.is_null() || parms.is_null() || ioarray.is_null() {
        return;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let parms_slice = slice::from_raw_parts_mut(parms, 15);
        let n = parms_slice[0] as usize;
        let p = parms_slice[1] as usize;
        let ntde = parms_slice[13] as usize;
        let p_total = p + ntde;
        let cards_slice = slice::from_raw_parts(cards, n * (2 * p + 4 + 2 * ntde));
        let ioarray_slice = slice::from_raw_parts_mut(ioarray, 9 * p_total);
        let data =
            native_data::NativeData::from_native_arrays(cards_slice, parms_slice, ioarray_slice, 9);
        profile::profile(&data, parms_slice, ioarray_slice);
    }));

    if result.is_err() {
        *parms.add(8) = 98.0;
        *parms.add(9) = -98.0;
    }
}
