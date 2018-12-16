use std::mem;

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub enum PositionType {
    Long,
    Short
}

impl PositionType {
    pub fn from_str(data: &str) -> PositionType {
        match data {
            "Long" => PositionType::Long,
            "Short" => PositionType::Short,
            o => panic!("Invalid position type {}!", o)
        }
    }

    pub fn opposite(&self) -> Self {
        if self == &PositionType::Long {
            PositionType::Short
        }
        else {
            PositionType::Long
        }
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub pos_type: PositionType,
    pub instrument: String,
    pub units: u32,
    pub avg_price: f64
}

#[derive(Debug, Clone)]
pub enum Granularity {
    M1,
    M5,
    M15,
    M30,
    D1,
    W1
}

impl Granularity {
    pub fn from_parts(time: char, freq: i32) -> Option<Granularity> {
        match time {
            'M' => {
                match freq {
                    1 => Some(Granularity::M1),
                    5 => Some(Granularity::M5),
                    15 => Some(Granularity::M15),
                    30 => Some(Granularity::M30),
                    _ => None
                }
            },
            'D' => {
                match freq {
                    1 => Some(Granularity::D1),
                    _ => None
                }
            },
            'W' => {
                match freq {
                    1 => Some(Granularity::W1),
                    _ => None
                }
            }
            _ => None
        }
    }

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

#[repr(C)]
pub struct History {
    pub bars: *const Bar,
    pub len: i32
}

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i32,
    pub spread: f64,
    #[serde(default)]
    pub date: i64,
    #[serde(default)]
    pub complete: bool
}

#[repr(C)]
pub struct RawString {
    pub raw: *mut u8,
    pub len: usize
}

impl RawString {
    pub fn overwrite(&mut self, data: String) {
        let mut bytes = data.into_bytes();
        self.raw = bytes.as_mut_ptr();
        self.len = bytes.len();
        mem::forget(bytes);
    }

    pub fn to_string(&self) -> String {
        unsafe {
            let dref = String::from_raw_parts(self.raw, self.len, self.len);
            let copy = dref.clone();
            mem::forget(dref);
            copy
        }
    }
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub id: String,
    pub date: i64,
    pub instrument: String,
    pub units: u32,
    pub position: PositionType,
    pub open_price: f64,
    pub closed: Vec<(f64, u32)>,
}

impl Trade {
    pub fn realized_profit(&self) -> f64 {
        self.closed.iter().map(|(price, units)| self.profit(price, units)).sum()
    }

    pub fn unrealized_profit(&self, bid: f64, ask: f64) -> f64 {
        let unclosed = self.unclosed_units();
        if unclosed > 0 {
            self.profit(&self.close_price(bid, ask), &unclosed)
        }
        else {
            0.0
        }
    }

    pub fn close(&mut self, price: f64, units: u32) -> (f64, u32) {
        let unclosed_units = self.unclosed_units();
        let to_close = if units > unclosed_units { unclosed_units } else { units };
        self.closed.push((price, to_close));
        (self.profit(&price, &to_close), to_close)
    }

    pub fn open(&self) -> bool {
        !self.closed()
    }

    pub fn closed(&self) -> bool {
        self.units == self.closed.iter().map(|(_, units)| units).sum::<u32>()
    }

    pub fn unclosed_units(&self) -> u32 {
        self.units - self.closed.iter().map(|(_, units)| units).sum::<u32>()
    }

    pub fn close_price(&self, bid: f64, ask: f64) -> f64 {
        match self.position {
            PositionType::Long => bid,
            PositionType::Short => ask
        }
    }

    fn profit(&self, close: &f64, units: &u32) -> f64 {
        (match self.position {
            PositionType::Long => close - self.open_price,
            PositionType::Short => self.open_price - close
        }) * (*units as f64)
    }
}

#[repr(C)]
pub struct TradeRequest {
    pub instrument: RawString,
    pub quantity: u32,
    pub position: RawString
}

impl TradeRequest {
    pub fn instrument(&self) -> String {
        self.instrument.to_string()
    }

    pub fn position(&self) -> String {
        self.position.to_string()
    }
}

#[repr(C)]
pub struct TradeResult {
    pub id: *const RawString,
    pub instrument: *const RawString,
    pub quantity: u32,
    pub position: *const RawString
}

#[derive(Debug, Clone)]
pub struct AccountInfo {
    pub balance: f64,
    pub unrealized_balance: f64,
    pub margin_available: f64,
    pub margin_rate: f64,
    pub margin_used: f64,
    pub unrealized_pl: f64
}
