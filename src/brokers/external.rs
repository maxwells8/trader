use super::*;
use datasource;
use brokers;
use session::{Session, RawSession, ToRaw};
use models::{TradeRequest, RawString, PositionType};

fn set_var<D: datasource::DataSource, B: brokers::Broker, F>
(sess: &Session<D, B>, var: *mut f64, func: F) where F: Fn(&B) -> f64 {
    unsafe {
        *var = func(&sess.broker.borrow())
    }
}

#[export]
fn current_balance(var: *mut f64) {
    set_var(sess, var, Broker::current_balance)
}

#[export]
fn current_price(var: *mut f64) {
    set_var(sess, var, Broker::current_price)
}

#[export]
fn unrealized_pl(var: *mut f64) {
    set_var(sess, var, Broker::unrealized_pl)
}

#[export]
fn unrealized_balance(var: *mut f64) {
    set_var(sess, var, Broker::unrealized_balance)
}

#[export]
fn unrealized_trade_pl(id: *mut u8, len: usize, var: *mut f64) {
    set_var(sess, var, move |b| {
        unsafe {
             b.unrealized_trade_pl(String::from_raw_parts(id, len, len))
        }
    })
}

#[export]
fn used_margin(var: *mut f64) {
    set_var(sess, var, Broker::used_margin)
}

#[export]
fn available_margin(var: *mut f64) {
    set_var(sess, var, Broker::available_margin)
}

#[export]
fn percent_change(var: *mut f64) {
    set_var(sess, var, Broker::percent_change)
}

#[export]
fn position_size(var: *mut i32) {
    unsafe {
        (*var) = sess.broker.borrow().position_size();
    }
}

#[export]
fn units_available(var: *mut u32) {
    unsafe {
        (*var) = sess.broker.borrow().units_available();
    }
}

#[export]
fn place_trade(request: *const TradeRequest, id: *mut RawString) {
    let request = unsafe { &*request };
    let res = sess.broker.borrow_mut().place_trade(request.instrument(), request.quantity, PositionType::from_str(&request.position()));
    if let Some(res) = res {
        unsafe {
            let id = &mut (*id);
            id.overwrite(res);
        };
    }
}

#[export]
fn close_trade(id: *const RawString, var: *mut f64) {
    unsafe {
        (*var) = sess.broker.borrow_mut().close_trade((*id).to_string());
    }
}

#[export]
fn close_units(units: u32, var: *mut f64) {
    unsafe {
        (*var) = sess.broker.borrow_mut().close_units(units);
    }
}
