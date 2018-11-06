from ctypes import *


class RawString(Structure):
    _fields_ = [
        ("raw", c_char_p),
        ("len", c_size_t)
    ]

def as_raw_str(val):
    c = c_char_p(bytes(val, 'utf-8'))
    return RawString(c, len(val))

class TradeRequest(Structure):
    _fields_ = [
        ("instrument", RawString),
        ("quantity", c_uint32),
        ("position", RawString)
    ]

class Bar(Structure):
    _fields_ = [
        ("open", c_double),
        ("high", c_double),
        ("low", c_double),
        ("close", c_double),
        ("volume", c_int),
        ("spread", c_double),
        ("date", c_long),
        ("complete", c_bool)
    ]

    def __repr__(self):
        return "{ \"open\": " + str(self.open) + ", \"high\": " + str(self.high) + ", \"low\": " + str(self.low) + ", \"close\": " + str(self.close) + ", \"volume\": " + str(self.volume) + ", \"date\": " + str(self.date) + " }"

class History(Structure):
    _fields_ = [("bars", POINTER(Bar)), ("len", c_int)]
