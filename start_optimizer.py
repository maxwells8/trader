import redis
import sys
import time
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')

# server_host = "192.168.0.115"
# server = redis.Redis(server_host)
#
# server.set("trajectory_steps", 6)
# server.set("gamma", 0.99)
#
# server.set("max_rho", 1)
# server.set("max_c", 1)
#
# server.set("actor_temp_cooldown", 0.999999)
# server.set("min_proposed", 0.001)
#
# server.set("critic_weight", 1)
# server.set("actor_v_weight", 1)
# server.set("actor_entropy_weight", 0.001)
# server.set("weight_penalty", 0.00001)
#
# server.set("learning_rate", 0.0001)
#
# server.set("queued_batch_size", 8)
# server.set("replay_buffer_size", 10000)
#
# server.set("reward_tau", 1e-6)
#
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
#
# import optimizer
# optimizer.Optimizer(dir_path + '/models/', server_host).run()


# -------------------------------------------------------------------
# server_host = "192.168.0.115"
# server = redis.Redis(server_host)
#
# server.set("queued_batch_size", 64)
# server.set("learning_rate", 1e-4)
# server.set("weight_penalty", 1e-4)
#
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
#
# import encoder_decoder_regressor
# encoder_decoder_regressor.Optimizer(dir_path + '/models/', server_host).run()

# -------------------------------------------------------------------
server_host = "192.168.0.115"
server = redis.Redis(server_host)

server.set("batch_size", 32)
server.set("learning_rate", 1e-4)
server.set("weight_penalty", 1e-4)
server.set("KL_coef", 1)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import encoder_decoder_classifier
encoder_decoder_classifier.Optimizer(dir_path + '/models/', server_host).run()
