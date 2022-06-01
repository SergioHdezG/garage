# Load the policy
import time

from garage.experiment import Snapshotter

snapshotter = Snapshotter()
data = snapshotter.load('data/local/experiment/maml_ppo_cnn_maze_pickle_dir')
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

steps, max_steps = 0, 299
done = False
obs, _ = env.reset()  # The initial observation
env.render('human')
policy.reset()

while steps < max_steps and not done:
    action, _ = policy.get_action(obs)
    result = env.step(action)
    env.render('human')  # Render the environment (optional)
    obs = result.observation
    steps += 1
    time.sleep(0.5)

env.close()
