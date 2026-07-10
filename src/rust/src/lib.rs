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
/// Run the optimizer from separate compact R arrays. Score weights are
/// implicitly one, matching every package-level caller, and are not copied.
///
/// # Safety
///
/// Array lengths must match the dimensions encoded in `parms`. `parms` and
/// `ioarray` must be writable; all other arrays are read-only.
pub unsafe extern "C" fn rust_firthcox_compact(
    x: *const f64,
    response: *const f64,
    start_order: *const i32,
    timedata: *const f64,
    parms: *mut f64,
    ioarray: *mut f64,
) {
    if x.is_null()
        || response.is_null()
        || start_order.is_null()
        || parms.is_null()
        || ioarray.is_null()
    {
        return;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let parms_slice = slice::from_raw_parts_mut(parms, 15);
        let n = parms_slice[0] as usize;
        let p = parms_slice[1] as usize;
        let ntde = parms_slice[13] as usize;
        assert!(ntde == 0 || !timedata.is_null());
        let p_total = p + ntde;
        let x_slice = slice::from_raw_parts(x, n * p);
        let response_slice = slice::from_raw_parts(response, n * 3);
        let start_order_slice = slice::from_raw_parts(start_order, n);
        let timedata_slice = if ntde == 0 {
            &[]
        } else {
            slice::from_raw_parts(timedata, n * ntde)
        };
        let ioarray_slice = slice::from_raw_parts_mut(ioarray, (3 + p_total) * p_total);
        let data = native_data::NativeData::from_compact_arrays(
            x_slice,
            response_slice,
            start_order_slice,
            timedata_slice,
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
        profile::profile(&data, parms_slice, ioarray_slice, None);
    }));

    if result.is_err() {
        *parms.add(8) = 98.0;
        *parms.add(9) = -98.0;
    }
}

#[no_mangle]
/// Compute profile likelihood intervals from separate compact R arrays.
///
/// # Safety
///
/// Array lengths must match the dimensions encoded in `parms`. `parms` and
/// `ioarray` must be writable; all other arrays are read-only.
pub unsafe extern "C" fn rust_plcomp_compact(
    x: *const f64,
    response: *const f64,
    start_order: *const i32,
    timedata: *const f64,
    parms: *mut f64,
    ioarray: *mut f64,
) {
    if x.is_null()
        || response.is_null()
        || start_order.is_null()
        || parms.is_null()
        || ioarray.is_null()
    {
        return;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let parms_slice = slice::from_raw_parts_mut(parms, 15);
        let n = parms_slice[0] as usize;
        let p = parms_slice[1] as usize;
        let ntde = parms_slice[13] as usize;
        assert!(ntde == 0 || !timedata.is_null());
        let p_total = p + ntde;
        let x_slice = slice::from_raw_parts(x, n * p);
        let response_slice = slice::from_raw_parts(response, n * 3);
        let start_order_slice = slice::from_raw_parts(start_order, n);
        let timedata_slice = if ntde == 0 {
            &[]
        } else {
            slice::from_raw_parts(timedata, n * ntde)
        };
        let ioarray_slice = slice::from_raw_parts_mut(ioarray, 9 * p_total);
        let data = native_data::NativeData::from_compact_arrays(
            x_slice,
            response_slice,
            start_order_slice,
            timedata_slice,
            parms_slice,
            ioarray_slice,
            9,
        );
        profile::profile(&data, parms_slice, ioarray_slice, None);
    }));

    if result.is_err() {
        *parms.add(8) = 98.0;
        *parms.add(9) = -98.0;
    }
}

#[no_mangle]
/// Run parameter estimation and profile likelihood using one converted data
/// set, avoiding a second copy of all observation-level arrays.
///
/// # Safety
///
/// Array lengths must match the dimensions encoded in the parameter arrays.
/// Both parameter and IO arrays must be writable.
pub unsafe extern "C" fn rust_fit_profile_compact(
    x: *const f64,
    response: *const f64,
    start_order: *const i32,
    timedata: *const f64,
    fit_parms: *mut f64,
    fit_ioarray: *mut f64,
    profile_parms: *mut f64,
    profile_ioarray: *mut f64,
    profile_selection: *const i32,
) {
    if x.is_null()
        || response.is_null()
        || start_order.is_null()
        || fit_parms.is_null()
        || fit_ioarray.is_null()
        || profile_parms.is_null()
        || profile_ioarray.is_null()
        || profile_selection.is_null()
    {
        return;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let fit_parms_slice = slice::from_raw_parts_mut(fit_parms, 15);
        let profile_parms_slice = slice::from_raw_parts_mut(profile_parms, 15);
        let n = fit_parms_slice[0] as usize;
        let p = fit_parms_slice[1] as usize;
        let ntde = fit_parms_slice[13] as usize;
        assert_eq!(profile_parms_slice[0] as usize, n);
        assert_eq!(profile_parms_slice[1] as usize, p);
        assert_eq!(profile_parms_slice[13] as usize, ntde);
        assert!(ntde == 0 || !timedata.is_null());
        let p_total = p + ntde;
        let fit_io_nrow = 3 + p_total;
        let x_slice = slice::from_raw_parts(x, n * p);
        let response_slice = slice::from_raw_parts(response, n * 3);
        let start_order_slice = slice::from_raw_parts(start_order, n);
        let timedata_slice = if ntde == 0 {
            &[]
        } else {
            slice::from_raw_parts(timedata, n * ntde)
        };
        let fit_ioarray_slice = slice::from_raw_parts_mut(fit_ioarray, fit_io_nrow * p_total);
        let profile_ioarray_slice = slice::from_raw_parts_mut(profile_ioarray, 9 * p_total);
        let profile_selection_slice = slice::from_raw_parts(profile_selection, p_total);
        let data = native_data::NativeData::from_compact_arrays(
            x_slice,
            response_slice,
            start_order_slice,
            timedata_slice,
            fit_parms_slice,
            fit_ioarray_slice,
            fit_io_nrow,
        );

        fit::fit(&data, fit_parms_slice, fit_ioarray_slice);
        for j in 0..p_total {
            profile_ioarray_slice[2 + 9 * j] = fit_ioarray_slice[2 + fit_io_nrow * j];
        }
        profile::profile(
            &data,
            profile_parms_slice,
            profile_ioarray_slice,
            Some(profile_selection_slice),
        );
    }));

    if result.is_err() {
        *fit_parms.add(7) = 98.0;
        *fit_parms.add(9) = -98.0;
        *profile_parms.add(8) = 98.0;
        *profile_parms.add(9) = -98.0;
    }
}
