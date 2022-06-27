from garage.experiment import Snapshotter
import matplotlib.pyplot as plt
from garage import rollout

snapshotter = Snapshotter()
data = snapshotter.load('/home/carlos/resultados/maml_ppo_resnet_maze_1', itr=31)
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

plt.imshow(env.render_top_view())
plt.show()

path = rollout(env, policy, animated=True)

print('Last reward: {}, Finished: {}, Termination: {}, Actions: {}'.format(
    path['rewards'],
    path['dones'][-1],
    path['env_infos']['TimeLimit.truncated'][-1],
    path['actions']
))

env.close()
