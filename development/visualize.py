# Load the policy
from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session

snapshotter = Snapshotter()
data = snapshotter.load('path/to/snapshot/dir')
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

steps, max_steps = 0, 150
done = False
obs = env.reset()  # The initial observation
policy.reset()

while steps < max_steps and not done:
    obs, rew, done, _ = env.step(policy.get_action(obs))
    env.render()  # Render the environment to see what's going on (optional)
    steps += 1

env.close()
