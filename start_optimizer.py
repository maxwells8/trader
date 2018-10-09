import redis
import sys
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer

server = redis.Redis("localhost")

server.set("gamma", 0.99)
server.set("optimizer_tau", 0.05)
server.set("optimizer_max_rho", 1)

server.set("optimizer_proposed_weight", 0.5)
server.set("optimizer_proposed_non_zero_weight", 0.05)
server.set("optimizer_critic_weight", 1)
server.set("optimizer_actor_weight", 0.5)
server.set("optimizer_entropy_weight", 0.25)
server.set("optimizer_weight_penalty", 0.01)

server.set("optimizer_learning_rate", 0.0001)

server.set("optimizer_prioritized_batch_size", 64)
server.set("optimizer_queued_batch_size", 32)

optimizer.Optimizer('C:\\Users\\Preston\\Programming\\trader\\models').run()
