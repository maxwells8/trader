use super::*;

mod fifo;

pub use self::fifo::FIFO;

pub trait TradeManager {
    fn open<'t, 's: 't>(&'s self) -> Vec<&'t Trade>;
    fn closed<'t, 's: 't>(&'s self) -> Vec<&'t Trade>;
    fn take_action<'c, C: 'c + Connector>(&mut self, connector: &'c C, action: Action, timestamp: i64);
    fn evaluate<'c, C: 'c + Connector, F>(&mut self, connector: &'c C, should_close: F) where F: Fn(&Trade) -> bool;

    fn display(&mut self, epoch: usize, current_tick: &Tick) -> String {
        let open = self.open();
        let closed = self.closed();
        let winnings = closed.iter().map(|trade| trade.pips().unwrap()).filter(|pips| pips.is_sign_positive()).collect::<Vec<f64>>();
        let losses = closed.iter().map(|trade| trade.pips().unwrap()).filter(|pips| pips.is_sign_negative()).collect::<Vec<f64>>();
        format!("{}: [{}] - {:.2} w/l; {} closed; {:.2} pips; {} open; {:.2} pips; {}; [{:.2}/{:.2}] avg pips;",
               Utc.timestamp(current_tick.timestamp, 0).to_rfc3339(),
               purple!(epoch),
               avg!(if closed.is_empty() { 0f64 } else { winnings.len() as f64 / closed.len() as f64 * 100. }),
               cyan!(closed.len()),
               win_loss!(closed.iter().map(|trade| trade.pips().unwrap()).sum::<f64>()),
               cyan!(open.len()),
               win_loss!(open.iter().map(|trade| calc_pips(trade.open, current_tick.close_price(&trade.action), &trade.action)).sum::<f64>()),
               cyan!(format!("{:?}", if open.len() > 0 { &open[0].action } else { &Action::None })),
               win_loss!(if winnings.is_empty() { 0f64 } else { winnings.iter().sum::<f64>() / winnings.len() as f64 }),
               win_loss!(if losses.is_empty() { 0f64 } else { losses.iter().sum::<f64>() / losses.len() as f64 })
        )
    }
}