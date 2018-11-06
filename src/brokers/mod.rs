pub mod sim;
pub mod external;
pub use self::sim::Sim;
use models::Bar;
use uuid::Uuid;

pub trait Broker {
    fn current_balance(&self) -> f64;
    fn current_price(&self) -> f64; //TODO Maybe Quote?
    fn unrealized_pl(&self) -> f64;
    fn unrealized_balance(&self) -> f64;
    fn unrealized_trade_pl(&self, id: String) -> f64;
    fn percent_change(&self) -> f64;

    fn place_trade(&mut self, instrument: String, units: u32, pos: Position) -> String;
    fn close_trade(&mut self, id: String) -> f64;

    fn on_bar(&mut self, bar: &Bar);
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub enum Position {
    Long,
    Short
}

impl Position {
    fn from_str(data: &str) -> Position {
        match data {
            "Long" => Position::Long,
            "Short" => Position::Short,
            o => panic!("Invalid position type {}!", o)
        }
    }
}

pub struct Trade {
    pub id: Uuid,
    pub date: i64,
    pub instrument: String,
    pub units: u32,
    pub position: Position,
    pub open_price: f64,
    pub close_price: Option<f64>
}

impl Trade {
    pub fn realized_profit(&self) -> Option<f64> {
        if let Some(price) = self.close_price {
            Some(self.profit(price))
        }
        else {
            None
        }
    }

    pub fn unrealized_profit(&self, bid: f64, ask: f64) -> f64 {
        self.profit(match self.position {
            Position::Long => bid,
            Position::Short => ask
        })
    }

    fn profit(&self, close: f64) -> f64 {
        (match self.position {
            Position::Long => close - self.open_price,
            Position::Short => self.open_price - close
        }) * (self.units as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profit() {
        let trade = Trade {
            id: Uuid::new_v4(),
            date: 0,
            instrument: String::from("EUR/USD"),
            units: 100,
            position: Position::Short,
            open_price: 1.15148,
            close_price: Some(1.15348)
        };
        println!("{:?}", trade.realized_profit())
    }
}