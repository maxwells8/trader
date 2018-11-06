use super::Broker;
use models::Bar;
use brokers::Trade;
use uuid::Uuid;
use brokers::Position;

pub struct Sim {
    current_time: i64,
    current_price: f64,
    spread: f64,
    trades: Vec<Trade>,
    end_balance: f64, //Balance after prev. bar ended
    balance: f64
}

impl Sim {
    pub fn new() -> Self {
        Sim {
            current_time: 0,
            current_price: 0.0,
            spread: 0.0,
            trades: vec![],
            end_balance: 2000.0,
            balance: 2000.0
        }
    }
}

impl Broker for Sim {
    fn current_balance(&self) -> f64 {
        self.balance
    }

    fn current_price(&self) -> f64 {
        self.current_price
    }

    fn unrealized_pl(&self) -> f64 {
        self.trades
            .iter()
            .filter(|trade| trade.close_price == None)
            .map(|trade| trade.unrealized_profit(self.current_price, self.current_price + self.spread))
            .sum()
    }

    fn unrealized_balance(&self) -> f64 {
        self.balance + self.unrealized_pl()
    }

    fn unrealized_trade_pl(&self, mut id: String) -> f64 {
        let uuid = Uuid::parse_str(&mut id).unwrap();
        for trade in &self.trades {
            if trade.id == uuid {
                if trade.close_price != None { panic!("The trade is already closed!") }
                return trade.unrealized_profit(self.current_price, self.current_price + self.spread);
            }
        }
        panic!("Trade Not Found!!")
    }

    fn percent_change(&self) -> f64 {
        (self.unrealized_balance() / self.end_balance) - 1.0
    }

    fn place_trade(&mut self, instrument: String, units: u32, pos: Position) -> String {
        let uuid = Uuid::new_v4();
        let id = uuid.to_string();
        self.trades.push(Trade {
            id: uuid,
            date: self.current_time,
            instrument,
            units,
            position: pos,
            open_price: match pos {
                Position::Long => self.current_price + self.spread,
                Position::Short => self.current_price
            },
            close_price: None,
        });
        id
    }

    fn close_trade(&mut self, mut id: String) -> f64 {
        if let Ok(uuid) = Uuid::parse_str(&mut id) {
            for trade in &mut self.trades {
                if trade.id == uuid {
                    trade.close_price = Some(match trade.position {
                        Position::Long => self.current_price,
                        Position::Short => self.current_price + self.spread
                    });
                    let pl = trade.realized_profit().unwrap();
                    self.balance += pl;
                    return pl;
                }
            }
        }
        panic!("Trade does not exist!")
    }

    fn on_bar(&mut self, bar: &Bar) {
        if self.current_time != 0 {
            self.end_balance = self.unrealized_balance();
        }
        self.current_time = bar.date;
        self.current_price = bar.close;
        self.spread = bar.spread;
    }
}

//TODO Write tests!!
