import redis
import sys
import time
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer
import encoder_decoder_classifier

server = redis.Redis("localhost")

server.set("trajectory_steps", 30)
server.set("gamma", 0.995)

server.set("max_rho", 10)
server.set("max_c", 1)

server.set("actor_temp_cooldown", 0.9999)
server.set("min_proposed", 0.001)

server.set("critic_weight", 1)
server.set("proposed_v_weight", 1)
server.set("proposed_entropy_weight", 0.001)
server.set("actor_v_weight", 1)
server.set("actor_entropy_weight", 0.1)
server.set("weight_penalty", 0.0001)

server.set("learning_rate", 0.00001)

server.set("prioritized_batch_size", 32)
server.set("queued_batch_size", 64)

server.set("reward_tau", 1e-6)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

optimizer.Optimizer(dir_path + '/models/').run()
# encoder_decoder_classifier.Optimizer(dir_path + '/models/').run()
