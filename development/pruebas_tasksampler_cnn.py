#!/usr/bin/env python3
"""This is an example to train MAML-PPO on Maze environment."""
# pylint: disable=no-value-for-parameter
import click
import torch
from gym_miniworld.envs import MazeS3Fast
import psutil

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler, VecWorker
from garage.torch.algos import MAMLPPO, MAMLTRPO
from garage.torch.policies import CategoricalCNNPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch import set_gpu_mode


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=300)
@click.option('--episodes_per_task', default=40)
@click.option('--meta_batch_size', default=20)
@wrap_experiment(snapshot_mode='all', log_dir='/home/carlos/resultados/',
                 prefix='experiments')
def maml_ppo_cnn_maze_dir(ctxt, seed, epochs, episodes_per_task,
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
    env = normalize(GymEnv(MazeS3Fast(),
                           is_image=True,
                           max_episode_length=max_episode_length))

    policy = CategoricalCNNPolicy(
        env_spec=env.spec,
        image_format='NHWC',
        hidden_nonlinearity=torch.relu,
        hidden_channels=(64, 32, 16),
        kernel_sizes=(4, 4, 3)
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None,
                                              is_image=True)

    task_sampler = SetTaskSampler(
        MazeS3Fast,
        wrapper=lambda env, _: normalize(GymEnv(
            env, is_image=True, max_episode_length=max_episode_length)))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=2,
                                   n_test_episodes=10)

    trainer = Trainer(ctxt)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         worker_class=VecWorker,
                         worker_args=dict(n_envs=12),
                         max_episode_length=env.spec.max_episode_length)

    algo = MAMLPPO(env=env,
                   policy=policy,
                   sampler=sampler,
                   task_sampler=task_sampler,
                   value_function=value_function,
                   meta_batch_size=meta_batch_size,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)

    # send policy to GPU
    if torch.cuda.is_available():
        device = set_gpu_mode(True)
        policy.to(device=device)

    # # Set tensorboard
    # tb = program.TensorBoard()
    # tb.configure(
    #     argv=[None, '--logdir', ctxt.snapshot_dir, '--host', '0.0.0.0'])
    # url = tb.launch()
    # print(f"Tensorflow listening on {url}")

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task)  # batch size no more than
    # 400 or 500 aprox due to RAM limitations (128GB)


maml_ppo_cnn_maze_dir()
