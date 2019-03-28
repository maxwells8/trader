import sys
sys.path.insert(0, "./worker")
from worker import Worker
# from gym_worker import Worker

def start_worker(name, instrument, granularity, server_host, start, test=False):
    new_worker = Worker(name, instrument, granularity, server_host, start, test)
    new_worker.run()
