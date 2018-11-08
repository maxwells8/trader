import redis
import sys
import time
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer
import encoder_optimizer

server = redis.Redis("localhost")

server.set("trajectory_steps", 32)
server.set("trajectory_steps", 120)
server.set("gamma", 0.99)

server.set("max_rho", 1)
server.set("max_c", 1)

server.set("proposer_tau", 0.01)
server.set("critic_tau", 0.01)
server.set("actor_v_tau", 0.01)
server.set("actor_pot_tau", 0.01)
server.set("entropy_tau", 0.01)
server.set("non_investing_tau", 0.01)
server.set("proposer_ema", None)
server.set("critic_ema", None)
server.set("actor_v_ema", None)
server.set("actor_pot_ema", None)
server.set("entropy_ema", None)

server.set("proposed_weight", 0.25)
server.set("critic_weight", 1)
server.set("actor_v_weight", 1)
server.set("entropy_weight", 0.05)
server.set("advantage_weight", 1)
server.set("weight_penalty", 0.01)

server.set("learning_rate", 0.0001)

server.set("prioritized_batch_size", 0)
server.set("queued_batch_size", 8)

server.set("samples_per_trajectory", 20)

server.set("reward_tau", 0.0001)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# optimizer.Optimizer(dir_path + '/models/').run()
encoder_optimizer.Optimizer(dir_path + '/models/').run()
