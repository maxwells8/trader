import redis
import sys
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer

server = redis.Redis("localhost")

server.set("gamma", 0.99)
server.set("optimizer_tau", 0.05)
server.set("optimizer_max_rho", 1)

server.set("optimizer_proposed_weight", 1)
server.set("optimizer_proposed_non_zero_weight", 0.01)
server.set("optimizer_critic_weight", 1)
server.set("optimizer_actor_weight", 1)
server.set("optimizer_entropy_weight", 0.05)
server.set("optimizer_weight_penalty", 0.005)

server.set("optimizer_batch_size", 16)

this_optimizer = optimizer.Optimizer('C:\\Users\\Preston\\Programming\\trader\\models')
this_optimizer.run()
