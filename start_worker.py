import redis
import sys
sys.path.insert(0, './worker')
import worker

server = redis.Redis("localhost")

i = 2

if i == 0:
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2015-1.109864962131578.csv"
    name = '0'
    models_loc = 'C:\\Users\\Preston\\Programming\\trader\\models'
    window = 256
    n_steps = 1000000
    test = False

    server.set("sigma_" + name, 0.01)

elif i == 1:
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2016-1.1071083227321519.csv"
    name = '1'
    models_loc = 'C:\\Users\\Preston\\Programming\\trader\\models'
    window = 256
    n_steps = 1000000
    test = False

    server.set("sigma_" + name, 0.05)

elif i == 2:
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    name = '2'
    models_loc = 'C:\\Users\\Preston\\Programming\\trader\\models'
    window = 256
    n_steps = 1000000
    test = False

    server.set("sigma_" + name, 0.1)

worker = worker.Worker(source, name, models_loc, window, n_steps, test)
worker.run()
