#!/usr/bin/env python3
"""This is an example to train MAML-PPO on Maze environment."""
# pylint: disable=no-value-for-parameter
import click
import torch
from gym_miniworld.envs import MazeS3Fast, MazeS3, Maze

from garage import wrap_experiment, _Default
from garage.envs import GymEnv, normalize
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler, VecWorker
from garage.torch.algos import MAMLPPO, MAMLTRPO, PPO
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import CategoricalCNNPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch import set_gpu_mode


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=30)
@click.option('--episodes_per_task', default=400)
@click.option('--meta_batch_size', default=5)
@wrap_experiment(snapshot_mode='all', log_dir='/home/carlos/resultados/',
                 prefix='experiments')
def maml_ppo_cnn_maze_no_meta(ctxt, seed, epochs, episodes_per_task,
                          meta_batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    set_seed(seed)
    max_episode_length = 300
    env = normalize(GymEnv(Maze(),
                           is_image=True,
                           max_episode_length=max_episode_length))

    policy = CategoricalCNNPolicy(
        env_spec=env.spec,
        image_format='NHWC',
        hidden_nonlinearity=torch.relu,
        hidden_channels=(128, 64, 32, 16),
        kernel_sizes=(5, 4, 4, 3)
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None,
                                              is_image=True)

    # task_sampler = SetTaskSampler(
    #     MazeS3Fast,
    #     wrapper=lambda env, _: normalize(GymEnv(
    #         env, is_image=True, max_episode_length=max_episode_length)))
    #
    # meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
    #                                n_test_tasks=10,
    #                                n_test_episodes=10)

    trainer = Trainer(ctxt)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         worker_class=VecWorker,
                         worker_args=dict(n_envs=12),
                         max_episode_length=env.spec.max_episode_length)

    policy_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=1e-1)), policy)
    vf_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=1e-1)),
        value_function)

    algo = PPO(env.spec,
               policy,
               value_function,
               sampler,
               policy_optimizer=policy_optimizer,
               vf_optimizer=vf_optimizer,
               lr_clip_range=5e-1,
               num_train_per_epoch=1,
               discount=0.99,
               gae_lambda=1.0,
               center_adv=True,
               positive_adv=False,
               policy_ent_coeff=0.0,
               use_softplus_entropy=False,
               stop_entropy_gradient=False,
               entropy_method='no_entropy')

    # send policy to GPU
    if torch.cuda.is_available():
        device = set_gpu_mode(True)
        policy.to(device=device)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task)  # batch size no more than
    # 400 or 500 aprox due to RAM limitations (128GB)


maml_ppo_cnn_maze_no_meta()
