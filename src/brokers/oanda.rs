use super::Broker;
use models::{Granularity, Bar, Trade, PositionType, Position, AccountInfo};
use connectors;

pub struct OANDA {
    client: connectors::oanda::OANDA,
    account_info: AccountInfo,
    position: Option<Position>,
    instrument: String,
    last_price: f64,
    change: f64
}

impl OANDA {
    pub fn new(instrument: String) -> Self {
        let client = connectors::oanda::OANDA::client();
        OANDA {
            account_info: client.fetch_account_info().expect("Account info is not currently available"),
            position: client.fetch_position(&instrument).expect("Position is not currently available"),
            last_price: client.load_latest_history(&instrument, &Granularity::M1, 1).last().map(|bar| bar.close).expect("Pricing is not currently available"),
            client,
            instrument,
            change: 0.0
        }
    }
}

impl Broker for OANDA {
    fn name() -> String {
        String::from("oanda")
    }

    fn current_balance(&self) -> f64 {
        self.account_info.balance
    }

    fn unrealized_balance(&self) -> f64 {
        self.account_info.unrealized_balance
    }

    fn current_price(&self) -> f64 {
        self.last_price
    }

    fn unrealized_pl(&self) -> f64 {
        self.account_info.unrealized_pl
    }

    fn unrealized_trade_pl(&self, id: String) -> f64 {
        unimplemented!()
    }

    fn percent_change(&self) -> f64 {
        self.change
    }

    fn used_margin(&self) -> f64 {
        self.account_info.margin_used
    }

    fn available_margin(&self) -> f64 {
        self.account_info.margin_available
    }

    fn position_size(&self) -> i32 {
        if let Some(ref pos) = self.position {
            pos.units as i32 * (if pos.pos_type == PositionType::Long { 1 } else { -1 })
        }
        else { 0 }
    }

    fn units_available(&self) -> u32 {
        ((self.available_margin() / self.account_info.margin_rate) / self.last_price) as u32
    }

    fn place_trade(&mut self, instrument: String, units: u32, pos: PositionType) -> Option<String> {
        if (self.client.submit_order(&self.instrument, &pos, units)) {
            self.position = self.client.fetch_position(&self.instrument).expect("Position is not currently available");
            self.account_info = self.client.fetch_account_info().expect("Account info is not currently available");
        }
        None
    }

    fn close_trade(&mut self, id: String) -> f64 {
        unimplemented!()
    }

    fn close_units(&mut self, units: u32) -> f64 {
        let bal = self.current_balance();
        let success = if let Some(ref pos) = self.position {
            self.client.submit_order(&self.instrument, &pos.pos_type.opposite(), units)
        }
        else { false };
        if success {
            self.position = self.client.fetch_position(&self.instrument).expect("Position is not currently available");
            self.account_info = self.client.fetch_account_info().expect("Account info is not currently available");

            return self.current_balance() - bal;
        }
        0.0
    }

    fn on_bar(&mut self, bar: &Bar) {
        let balance = self.unrealized_balance();
        self.last_price = bar.close;
        self.account_info = self.client.fetch_account_info().expect("Account info is not currently available");
        self.position = self.client.fetch_position(&self.instrument).expect("Position is not currently available");
        self.change = (self.unrealized_balance() / balance) - 1.0
    }
}
