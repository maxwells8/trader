import redis
import sys
import time
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer
import encoder_decoder_regressor

server = redis.Redis("localhost")

server.set("trajectory_steps", 10)
server.set("gamma", 0.99)

server.set("max_rho", 1)
server.set("max_c", 1)

server.set("actor_temp_cooldown", 0.99999)
server.set("min_proposed", 0.001)

server.set("critic_weight", 1)
server.set("actor_v_weight", 1)
server.set("actor_entropy_weight", 0.001)
server.set("weight_penalty", 0.0001)

server.set("learning_rate", 0.0001)

server.set("queued_batch_size", 64)
server.set("replay_buffer_size", 10000)

server.set("reward_tau", 1e-6)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

optimizer.Optimizer(dir_path + '/models/').run()
# encoder_decoder_regressor.Optimizer(dir_path + '/models/').run()
