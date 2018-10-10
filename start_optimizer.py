import redis
import sys
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
import optimizer

server = redis.Redis("localhost")

server.set("trajectory_steps", 100)
server.set("gamma", 0.99)

server.set("optimizer_tau", 0.05)
server.set("optimizer_max_rho", 1)

server.set("optimizer_proposed_weight", 1)
server.set("optimizer_critic_weight", 1)
server.set("optimizer_actor_weight", 1)
server.set("optimizer_entropy_weight", 0.25)
server.set("optimizer_weight_penalty", 0.05)

server.set("optimizer_learning_rate", 0.0001)

server.set("optimizer_prioritized_batch_size", 32)
server.set("optimizer_queued_batch_size", 16)

optimizer.Optimizer('C:\\Users\\Preston\\Programming\\trader\\models').run()
