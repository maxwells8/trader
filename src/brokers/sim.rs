use super::Broker;
use models::{Bar, PositionType, Trade};
use uuid::Uuid;
use chrono::{DateTime, TimeZone, Utc, Local, Weekday, Datelike, Timelike};
use utils::*;

pub struct Sim {
    current_time: i64,
    current_price: f64,
    spread: f64,
    trades: Vec<Trade>,
    end_balance: f64, //Balance after prev. bar ended
    balance: f64,
    margin_req: f64
}

impl Sim {
    pub fn new(instrument: &String) -> Self {
        let margin_req = match instrument.as_ref() {
            "EUR_USD" => 0.02,
            "GBP_USD" => 0.05,
            "AUD_USD" => 0.03,
            "NZD_USD" => 0.03,
            pair => panic!("{} is not a supported currency pair!", pair)
        };
        Sim {
            current_time: 0,
            current_price: 0.0,
            spread: 0.0,
            trades: vec![],
            end_balance: 2000.0,
            balance: 2000.0,
            margin_req,
        }
    }

    fn mid_point(&self) -> f64 {
        return self.current_price + (self.spread / 2.0);
    }


    fn unrealized_pl_midpoint(&self) -> f64 {
        let midpoint = self.mid_point();
        self.trades
            .iter()
            .map(|trade| trade.unrealized_profit(midpoint, midpoint))
            .sum()
    }

    fn unrealized_balance_midpoint(&self) -> f64 {
        self.balance + self.unrealized_pl_midpoint()
    }
}

impl Broker for Sim {
    fn name() -> String {
        String::from("sim")
    }

    fn current_balance(&self) -> f64 {
        self.balance
    }

    fn current_price(&self) -> f64 {
        self.current_price
    }

    fn unrealized_pl(&self) -> f64 {
        self.trades
            .iter()
            .map(|trade| trade.unrealized_profit(self.current_price, self.current_price + self.spread))
            .sum()
    }

    fn unrealized_balance(&self) -> f64 {
        self.balance + self.unrealized_pl()
    }

    fn unrealized_trade_pl(&self, mut id: String) -> f64 {
        for trade in &self.trades {
            if trade.id == id {
                return trade.unrealized_profit(self.current_price, self.current_price + self.spread);
            }
        }
        panic!("Trade Not Found!!")
    }

    fn used_margin(&self) -> f64 {
        let units = self.trades.iter().map(|trade| trade.unclosed_units()).sum::<u32>();
        (units as f64) * self.margin_req * self.mid_point()
    }

    fn available_margin(&self) -> f64 {
        (self.unrealized_balance_midpoint() - self.used_margin()).max(0.0)
    }

    fn percent_change(&self) -> f64 {
        (self.unrealized_balance() / self.end_balance) - 1.0
    }

    fn position_size(&self) -> i32 {
        let mut size = 0;
        for trade in &self.trades {
            let units = trade.unclosed_units();
            if units > 0 {
                if trade.position == PositionType::Long {
                    size += trade.unclosed_units() as i32
                } else {
                    size -= trade.unclosed_units() as i32
                }
            }
        }
        size
    }

    fn units_available(&self) -> u32 {
        ((self.available_margin() / self.margin_req) / self.current_price) as u32
    }

    fn place_trade(&mut self, instrument: String, units: u32, pos: PositionType) -> Option<String> {
        if !market_open(self.current_time) {
            println!("Failed to place trade! Trading is not currently allowed.");
            return None
        }

        if self.trades.is_empty() || self.trades.iter().all(|trade| trade.closed()) || self.trades.iter().filter(|trade| trade.open()).any(|trade| trade.position == pos) {
            let margin = (units as f64) * self.margin_req * self.mid_point();
            let available = self.available_margin();
            if margin <= available {
                let id = Uuid::new_v4().to_string();
                self.trades.push(Trade {
                    id: id.clone(),
                    date: self.current_time,
                    instrument,
                    units,
                    position: pos,
                    open_price: match pos {
                        PositionType::Long => self.current_price + self.spread,
                        PositionType::Short => self.current_price
                    },
                    closed: vec![],
                });
                return Some(id)
            }
            else {
                println!("Failed to place trade with margin ${}. Available margin is ${}", margin, available);
            }
        }
        else {
            println!("Failed to place trade with pos {:?}. Hedging is not allowed!", pos);
        }
        None
    }

    fn close_trade(&mut self, mut id: String) -> f64 {
        if !market_open(self.current_time) {
            println!("Failed to close trade! Trading is not currently allowed.");
            return 0.0
        }

        let bid = self.current_price;
        let ask = self.current_price + self.spread;
        let pl = self.trades.iter_mut().find(|trade| trade.id == id).map(|trade| {
            let price = trade.close_price(bid, ask);
            let units = trade.units;
            trade.close(price, units).0
        }).unwrap_or(0.0);
        self.balance += pl;
        pl
    }

    fn close_units(&mut self, units: u32) -> f64 {
        if !market_open(self.current_time) {
            println!("Failed to close trade! Trading is not currently allowed.");
            return 0.0
        }

        let mut units_left = units;
        let mut profit = 0.0;
        for trade in &mut self.trades {
            let price = trade.close_price(self.current_price, self.current_price + self.spread);
            let (pl, closed) = trade.close(price, units_left);
            self.balance += pl;
            profit += pl;
            units_left -= closed;
        }
        profit
    }

    fn on_bar(&mut self, bar: &Bar) {
        if self.current_time != 0 {
            self.end_balance = self.unrealized_balance();
        }
        self.current_time = bar.date;
        self.current_price = bar.close;
        self.spread = bar.spread;

        // Margin closeout
        if self.unrealized_balance_midpoint() <= self.used_margin() / 2.0 {
            let units_to_close = self.trades.iter().map(|trade| trade.unclosed_units()).sum::<u32>();
            self.close_units(units_to_close);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let sim = Sim::new(&String::from("EUR_USD"));
        sim.place_trade()
    }
}

//TODO Write tests!!
