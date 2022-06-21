from garage.experiment import Snapshotter
import matplotlib.pyplot as plt
from garage import rollout

snapshotter = Snapshotter()
data = snapshotter.load('/home/carlos/resultados/maml_ppo_cnn_maze_2')
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

plt.imshow(env.render_top_view())
plt.show()

path = rollout(env, policy, animated=True)

print('Last reward: {}, Finished: {}, Termination: {}'.format(
    path['rewards'][-1],
    path['dones'][-1],
    path['env_infos']['TimeLimit.truncated'][-1]
))

env.close()
