import sys
sys.path.insert(0, "./worker")
import simple_worker_regressor

def start_worker(instrument, granularity, server_host, start):
    new_worker = simple_worker_regressor.Worker(instrument, granularity, server_host, start)
    new_worker.run()
