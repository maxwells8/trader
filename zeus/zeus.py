from ctypes import *
from .data import *
import time
import sys
import codecs

def slashescape(err):
    """ codecs error handler. err is UnicodeDecode instance. return
    a tuple with a replacement for the unencodable part of the input
    and a position where encoding should continue"""
    #print err, dir(err), err.start, err.end, err.object[:err.start]
    thebyte = err.object[err.start:err.end]
    repl = u'\\x'+hex(ord(thebyte))[2:]
    return (repl, err.end)

codecs.register_error('slashescape', slashescape)

class Zeus:
    def __init__(self, instrument, granularity):
        self.instrument = instrument;
        self.granularity = granularity;

        self.lib = windll.LoadLibrary('target\debug\zeus.dll')

        self.lib.create_session.argtypes = [POINTER(RawString), c_char, c_int]
        self.lib.create_session.restype = c_void_p
        self.sess = self.lib.create_session(as_raw_str(instrument), c_char(bytes(granularity[0], 'utf-8')), int(granularity[1:]))

        #  Data
        self.lib.stream_bars.argtypes = [c_void_p, c_size_t, c_void_p]
        self.lib.stream_bars.restype = c_void_p
        self.lib.stream_range.argtypes = [c_void_p, c_long, c_long, c_void_p]
        self.lib.stream_range.restype = c_void_p
        # TODO self.lib.load_history.restype = c_void_p

        #  Broker Info
        self.lib.current_balance.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.current_balance.restype = c_void_p
        self.lib.current_price.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.current_price.restype = c_void_p
        self.lib.unrealized_pl.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.unrealized_pl.restype = c_void_p
        self.lib.unrealized_balance.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.unrealized_balance.restype = c_void_p
        self.lib.percent_change.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.percent_change.restype = c_void_p
        # TODO self.lib.unrealized_trade_pl.restype = c_void_p

        #  Trading
        self.lib.place_trade.argtypes = [c_void_p, POINTER(TradeRequest), POINTER(RawString)]
        self.lib.place_trade.restype = c_void_p
        self.lib.close_trade.argtypes = [c_void_p, POINTER(RawString), POINTER(c_double)]
        self.lib.close_trade.restype = c_void_p

    def _get_value(self, func):
        temp = c_double(0.0)
        self.sess = func(self.sess, byref(temp))
        return temp.value

    def current_balance(self):
        return self._get_value(self.lib.current_balance)

    def unrealized_pl(self):
        return self._get_value(self.lib.unrealized_pl)

    def unrealized_balance(self):
        return self._get_value(self.lib.unrealized_balance)

    def percent_change(self):
        return self._get_value(self.lib.percent_change)

    def stream_bars(self, count, callback):
        def on_bar(sess, bar, is_done):
            self.sess = sess
            callback(bar.contents)
            sys.stdout.flush()
            return self.sess
        BARFUNC = WINFUNCTYPE(c_void_p, c_void_p, POINTER(Bar), c_bool)(on_bar)
        self.sess = self.lib.stream_bars(self.sess, count, byref(BARFUNC))

    def stream_range(self, from_epoch, to_epoch, callback):
        def on_bar(sess, bar, is_done):
            self.sess = sess
            callback(bar.contents)
            sys.stdout.flush()
            return self.sess
        BARFUNC = WINFUNCTYPE(c_void_p, c_void_p, POINTER(Bar), c_bool)(on_bar)
        self.sess = self.lib.stream_range(self.sess, from_epoch, to_epoch, byref(BARFUNC))

    def place_trade(self, quantity, pos):
        req = TradeRequest(as_raw_str(self.instrument), quantity, as_raw_str(pos))
        id = as_raw_str("")
        self.sess = self.lib.place_trade(self.sess, byref(req), byref(id))
        return id.raw.decode('utf-8', 'ignore')[:id.len]

    def close_trade(self, id):
        id = as_raw_str(id)
        var = c_double(0.0)
        self.sess = self.lib.close_trade(self.sess, byref(id), byref(var))
        return var.value
