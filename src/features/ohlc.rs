use super::*;
use chrono::{Utc, Datelike, Timelike, DateTime, TimeZone};

#[derive(Clone)]
pub struct OHLC {
    name: String,
    min_pips: f64,
    max_loss: f64,
    max_duration: i64,
    sample_size: usize,
    trading_hours: Option<TradingHours>
}

impl Display for OHLC {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "ohlc_{}", self.name())
    }
}

impl OHLC {
    pub fn new(label: &str, min_pips: f64, max_loss: f64, max_duration: i64, sample_size: usize, trading_hours: Option<TradingHours>) -> Self {
        let hours = if trading_hours.is_some() {
            let th = trading_hours.as_ref().unwrap();
            format!("{}-{}", th.start, th.end)
        } else {
            String::from("None")
        };
        let name = format!("{}_{}_{}_{}_{}_{}", label, min_pips, max_loss, max_duration, sample_size, hours);
        OHLC {
            name,
            min_pips,
            max_loss,
            max_duration,
            sample_size,
            trading_hours
        }
    }

    pub fn name<'r, 's: 'r>(&'s self) -> &'r str {
        &self.name
    }

    pub fn sample_size(&self) -> usize {
        self.sample_size
    }

    pub fn can_trade(&self, bars: &Vec<Bar>) -> bool {
        if let Some(ref last) = bars.last() {
            if TradingHours::can_trade_at(&self.trading_hours, Utc.timestamp(last.date, 0)) {
                return true;
            }
        }

        false
    }
}

impl Features for OHLC {
    fn compile_input(&self, bars: &Vec<Bar>) -> Option<Vec<Vec<f64>>> {
        if self.can_trade(bars) {
            bars.last_sample(self.sample_size).map(|sample| sample.normalize())
        }
        else {
            None
        }
    }

    fn should_close(&self, trade: &Trade, tick: &Tick) -> bool {
        let pips = calc_pips(trade.open, tick.close_price(&trade.action), &trade.action);

        pips >= self.min_pips || tick.timestamp - trade.timestamp >= self.max_duration || pips <= self.max_loss
    }

    fn choose(&self, first: &Trade, second: &Trade) -> Action {
        let f_pip = first.pips().expect("Expected first trade to have already been closed!");
        let s_pip = second.pips().expect("Expected second trade to have already been closed!");
        if f_pip > s_pip && f_pip >= self.min_pips {
            first.action.clone()
        }
        else if s_pip >= self.min_pips {
            second.action.clone()
        }
        else {
            Action::None
        }
    }
}