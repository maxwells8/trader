use super::*;
use super::ohlc::OHLC;

#[derive(Clone)]
pub struct PercentChange {
    parent: OHLC
}

impl Display for PercentChange {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "percent_{}", self.parent.name())
    }
}

impl PercentChange {
    pub fn new(label: &str, min_pips: f64, max_loss: f64, max_duration: i64, sample_size: usize, trading_hours: Option<TradingHours>) -> Self {
        PercentChange {
            parent: OHLC::new(label, min_pips, max_loss, max_duration, sample_size, trading_hours)
        }
    }
}

impl Features for PercentChange {
    fn compile_input(&self, bars: &Vec<Bar>) -> Option<Vec<Vec<f64>>> {
        if self.parent.can_trade(bars) {
            bars.last_sample(self.parent.sample_size() + 1).map(|sample| {
                let mut new = Vec::<Bar>::new();
                for i in 1..sample.len() {
                    let prev = &sample[i - 1];
                    let cur = &sample[i];
                    let bar = Bar {
                        open: (cur.open - prev.open) / cur.open,
                        high: (cur.high - prev.high) / cur.high,
                        low: (cur.low - prev.low) / cur.low,
                        close: (cur.close - prev.close) / cur.close,
                        volume: (cur.volume - prev.volume) / cur.volume,
                        complete: cur.complete,
                        date: cur.date
                    };
                    new.push(bar);
                }
                new.normalize()
            })
        }
        else {
            None
        }
    }

    fn should_close(&self, trade: &Trade, tick: &Tick) -> bool {
        self.parent.should_close(trade, tick)
    }

    fn choose(&self, first: &Trade, second: &Trade) -> Action {
        self.parent.choose(first, second)
    }
}