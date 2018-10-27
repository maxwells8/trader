import sys
sys.path.insert(0, "./worker")
import simple_worker

def start_worker(source, name, start, n_steps):
    new_worker = simple_worker.Worker(source, name, start, n_steps)
    new_worker.run()
