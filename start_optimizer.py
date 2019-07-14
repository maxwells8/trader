import redis
import sys
import time
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')

# server_host = "192.168.0.115"
# server = redis.Redis(server_host)
#
# server.set("trajectory_steps", 5)
# server.set("gamma", 0.99)
#
# server.set("max_rho", 1)
# server.set("max_c", 1)
#
# server.set("actor_temp_cooldown", 0.999999)
#
# server.set("critic_weight", 1)
# server.set("actor_v_weight", 1)
# server.set("actor_entropy_weight", 1e-4)
# server.set("weight_penalty", 1e-5)
#
# server.set("learning_rate", 1e-5)
#
# server.set("queued_batch_size", 32)
# server.set("replay_buffer_size", 10000)
#
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
#
# import optimizer
# optimizer.Optimizer(dir_path + '/models/', server_host).run()
#
# 
# -------------------------------------------------------------------
server_host = "192.168.0.115"
server = redis.Redis(server_host)

server.set("queued_batch_size", 64)
server.set("learning_rate", 1e-4)
server.set("weight_penalty", 1e-4)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import encoder_decoder_regressor
encoder_decoder_regressor.Optimizer(dir_path + '/models/', server_host).run()

# -------------------------------------------------------------------
# server_host = "192.168.0.115"
# server = redis.Redis(server_host)
#
# server.set("batch_size", 32)
# server.set("learning_rate", 1e-4)
# server.set("weight_penalty", 1e-4)
# server.set("KL_coef", 1)
#
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
#
# import encoder_decoder_classifier
# encoder_decoder_classifier.Optimizer(dir_path + '/models/', server_host).run()
