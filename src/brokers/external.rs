use super::*;
use datasource;
use brokers;
use session::{Session, RawSession};
use models::{TradeRequest, RawString};

fn set_var<D: datasource::DataSource + Send + Sync + 'static, B: brokers::Broker + Send + Sync + 'static, F>
(sess: &Session<D, B>, var: *mut f64, func: F) where F: Fn(&B) -> f64 {
    unsafe {
        *var = func(&sess.broker.borrow())
    }
}

export!(current_balance(var: *mut f64) move |sess| {
    set_var(sess, var, Broker::current_balance)
});

export!(current_price(var: *mut f64) move |sess| {
    set_var(sess, var, Broker::current_price)
});

export!(unrealized_pl(var: *mut f64) move |sess| {
    set_var(sess, var, Broker::unrealized_pl)
});

export!(unrealized_balance(var: *mut f64) move |sess| {
    set_var(sess, var, Broker::unrealized_balance)
});

export!(unrealized_trade_pl(id: *mut u8, len: usize, var: *mut f64) move |sess| {
    set_var(sess, var, move |b| {
        unsafe {
             b.unrealized_trade_pl(String::from_raw_parts(id, len, len))
        }
    })
});

export!(percent_change(var: *mut f64) move |sess| {
    set_var(sess, var, Broker::percent_change)
});

export!(place_trade(request: *const TradeRequest, id: *mut RawString) move |sess| {
    let request = unsafe { &*request };
    let res = sess.broker.borrow_mut().place_trade(request.instrument(), request.quantity, Position::from_str(&request.position()));
    unsafe {
        let id = &mut (*id);
        id.overwrite(res);
    };
});

export!(close_trade(id: *const RawString, var: *mut f64) move |sess| {
    unsafe {
        (*var) = sess.broker.borrow_mut().close_trade((*id).to_string());
    }
});
