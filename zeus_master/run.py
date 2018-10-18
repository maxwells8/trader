#!/usr/bin/python3

from ctypes import *
import numpy as np
import numpy.ctypeslib as nc

from python.wrapper.zeus import Zeus

def do(bar):
    global zeus
    id = zeus.place_trade(1000, "Long")
    print(zeus.unrealized_pl())
    print(zeus.close_trade(id))
    print(zeus.percent_change())
    print()

zeus = Zeus("EUR_USD", "M1")
zeus.stream_bars(10, do)
print(zeus.unrealized_balance())
# zeus.stream_range(1538658304, 1538688304, lambda bar: print(bar.contents))

'''
zeus = cdll.LoadLibrary('target/debug/libzeus.so')

zeus.create_session.restype = c_void_p
sess = zeus.create_session("EUR_USD", "M", 1)

zeus.test.restype = c_void_p
zeus.test.argtypes = [c_void_p]
sess = zeus.test(sess)

print("-----------------------")

zeus.load_history.argtypes = [c_void_p, c_int, POINTER(History)]
zeus.load_history.restype = c_void_p
hist = History()
sess = zeus.load_history(sess, 10, byref(hist))

for i in range(hist.len):
    print(hist.bars[i])

print("-----------------------")

zeus.backtest.argtypes = [c_void_p, c_int, c_void_p]
zeus.backtest.restype = c_void_p
sess = zeus.backtest(sess, 1500, byref(on_tick))

while DONE is False:
    pass
'''
# for _ in range(10):
#     h = History()
#     sess = zeus.handle(sess, byref(h))
#     print(nc.as_array(h.bars, shape=(h.len, 3)))

#zeus.get_history.argtypes = [c_char_p, c_char_p, c_int]
# zeus.get_history.restype = History
# his = zeus.get_history("EUR_USD", "M", 15, 100000)
# print(his.len)
# bars = nc.as_array(his.bars, shape=(his.len, 6))
# print(bars)
