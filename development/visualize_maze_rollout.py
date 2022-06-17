# Load the policy
import time

from garage.experiment import Snapshotter

snapshotter = Snapshotter()
data = snapshotter.load('/home/carlos/resultados/maml_ppo_cnn_maze')
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

from garage import rollout
path = rollout(env, policy, animated=True, pause_per_frame=0.1)

env.close()
