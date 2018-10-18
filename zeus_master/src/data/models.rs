use super::super::Action;
use chrono::{Utc, Datelike, Timelike, DateTime, TimeZone};

#[derive(Serialize, Deserialize, Debug)]
pub struct Series {
    pub symbol: String,
    pub granularity: Granularity,
    pub bars: Vec<Bar>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Granularity {
    M1,
    M5,
    M15,
    M30,
    D1,
    W1
}

impl Granularity {
    pub fn minutes(&self) -> u32 {
        match self {
            &Granularity::M1 => 1,
            &Granularity::M5 => 5,
            &Granularity::M15 => 15,
            &Granularity::M30 => 30,
            &Granularity::D1 => 1440,
            &Granularity::W1 => 10080,
            _ => panic!("Invalid Granularity!")
        }
    }

    pub fn seconds(&self) -> u32 {
        self.minutes() * 60
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i32,
    #[serde(default)]
    pub complete: bool,
    #[serde(default)]
    pub date: i64
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tick {
    pub bid: f64,
    pub ask: f64,
    pub timestamp: i64
}

impl Tick {
    pub fn close_price(&self, action: &Action) -> f64 {
        match action {
            &Action::Buy => self.bid,
            &Action::Sell => self.ask,
            _ => 0.0
        }
    }
}

#[derive(Debug)]
pub struct Trade {
    pub open: f64,
    pub close: Option<f64>,
    pub timestamp: i64,
    pub action: Action
}

impl Trade {
    pub fn closed(&self) -> bool {
        self.close.as_ref().is_some()
    }

    pub fn pips(&self) -> Option<f64> {
        self.close.as_ref().map(|price| super::utils::calc_pips(self.open, *price, &self.action))
    }
}

#[derive(Clone, Debug)]
pub struct TradingHours {
    pub start: u32,
    pub end: u32
}

impl TradingHours {
    pub fn can_trade(hours: &Option<Self>) -> bool {
        TradingHours::can_trade_at(hours, Utc::now())
    }

    pub fn can_trade_at(hours: &Option<Self>, date: DateTime<Utc>) -> bool {
        if let &Some(ref hours) = hours {
            let hour = date.time().hour();
            let dow = date.date().weekday().number_from_monday();
            hour >= hours.start && hour <= hours.end && dow <= 5
        }
        else {
            true
        }
    }
}