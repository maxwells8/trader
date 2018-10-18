use super::*;
use super::super::Action;
use data::models::{Bar, Trade, Tick};
use data::utils::*;
use features::{Features, Bars};
use std::process::{Command, ExitStatus};
use json;
use std::thread::JoinHandle;
use std::thread;
use std::path::Path;
use num_cpus::get as get_num_cpu;

use std::ffi::CStr;
use libc::{c_double, c_schar, c_int};
use std::mem;

#[repr(C)]
pub struct Data {
    x: *const X,
    y: *const Y,
}

#[repr(C)]
pub struct X {
    values: *const *const *const f64,
    test: *const f64,
    len: i32,
    sample_size: i32,
    features: i32,
}

#[repr(C)]
pub struct Y {
    values: *const *const i32,
    len: i32,
    classifications: i32
}

pub struct ExternalTrainer {
    train_fn: unsafe extern fn(*const Data)
}

impl ExternalTrainer {
    pub fn new(func: unsafe extern fn(*const Data)) -> Self {
        ExternalTrainer { train_fn: func }
    }

    fn close(&self, trade: &mut Trade, tick: &Tick) -> bool {
        false
    }
}

fn to_ptr<R, T, M>(t: Vec<T>) -> *const R {
    if t[0]
    t.iter().map(|o| to_ptr(o)).collect::<Vec<*const M>>().as_ptr()
}

impl<'c, C: 'c + Connector> Trainer<'c, C> for ExternalTrainer {
    fn train(&mut self, connector: &'c C) {
        let test = vec![vec![vec![1.0, 46.3, 33.7]]];
        let x = Box::new(X {
            values: to_ptr(test),
            test: vec![1.1, 46.3, 33.7].as_ptr(),
            len: 1,
            sample_size: 1,
            features: 3
        });
        let y = Box::new(Y {
            values: vec![vec![5, 0].as_ptr()].as_ptr(),
            len: 1,
            classifications: 2
        });
        let data = Box::new(Data {
            x: &*x,
            y: &*y
        });
        unsafe {
            println!("{:?}", x.values.to_owned());
            let data_ptr: *const Data = &*data;
            mem::forget(test);
            mem::forget(x);
            mem::forget(y);
            mem::forget(data);
            (self.train_fn)(data_ptr);
        }
    }
}