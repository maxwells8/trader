use std::mem;

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