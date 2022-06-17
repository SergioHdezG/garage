# Load the policy
import time

from garage.experiment import Snapshotter

snapshotter = Snapshotter()
data = snapshotter.load('/home/carlos/resultados/maml_ppo_cnn_maze_scaled_2')
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']
max_steps = env.spec.max_episode_length

steps = 0
done = False
obs, _ = env.reset()  # The initial observation
env.render('human')
policy.reset()

while steps < max_steps and not done:
    action, _ = policy.get_action(obs)
    result = env.step(action)
    print(result.terminal, result.timeout, result.reward)
    env.render('human')  # Render the environment (optional)
    obs = result.observation
    steps += 1
    time.sleep(0.1)

env.close()
