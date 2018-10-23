import redis
import sys
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer

server = redis.Redis("localhost")

server.set("trajectory_steps", 120)
server.set("gamma", 0.99)

server.set("max_rho", 1)
server.set("max_c", 1)

server.set("proposer_tau", 0.01)
server.set("critic_tau", 0.01)
server.set("actor_tau", 0.01)
server.set("entropy_tau", 0.01)
server.set("non_investing_tau", 0.01)
server.set("proposer_ema", None)
server.set("critic_ema", None)
server.set("actor_ema", None)
server.set("entropy_ema", None)
server.set("non_investing_ema", None)

server.set("proposed_weight", 0.25)
server.set("critic_weight", 1)
server.set("actor_weight", 1)
server.set("entropy_weight", 0.1)
server.set("non_investing_weight", 0.5)
server.set("weight_penalty", 0.01)

server.set("learning_rate", 0.0001)

server.set("prioritized_batch_size", 0)
server.set("queued_batch_size", 4)

server.set("reward_tau", 0.0001)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

while True:
    # try:
    optimizer.Optimizer(dir_path + '/models/').run()
    # except Exception:
    #     print("ERROR IN OPTIMIZER -- RESTARTING")
