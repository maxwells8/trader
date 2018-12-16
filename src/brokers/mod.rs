pub mod sim;
pub mod oanda;
pub mod external;
pub use self::sim::Sim;
use models::{Bar, Trade, PositionType};
use uuid::Uuid;

pub trait Broker {
    fn name() -> String;
    fn current_balance(&self) -> f64;
    fn current_price(&self) -> f64;
    fn unrealized_pl(&self) -> f64;
    fn unrealized_balance(&self) -> f64;
    fn unrealized_trade_pl(&self, id: String) -> f64;
    fn percent_change(&self) -> f64;
    fn used_margin(&self) -> f64;
    fn available_margin(&self) -> f64;
    fn position_size(&self) -> i32;
    fn units_available(&self) -> u32;

    fn place_trade(&mut self, instrument: String, units: u32, pos: PositionType) -> Option<String>;
    fn close_trade(&mut self, id: String) -> f64;
    fn close_units(&mut self, units: u32) -> f64;

    fn on_bar(&mut self, bar: &Bar);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profit() {
        let trade = Trade {
            id: Uuid::new_v4().to_string(),
            date: 0,
            instrument: String::from("EUR/USD"),
            units: 100,
            position: PositionType::Short,
            open_price: 1.15148,
            closed: vec![(1.15348, 100)]
        };
        println!("{:?}", trade.realized_profit())
    }
}
