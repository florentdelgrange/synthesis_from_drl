import math
import os
import random
import sys

import numpy as np
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.typing.types import Float, FloatOrReturningFloat

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

from typing import Tuple, Callable, Optional, List
from collections import OrderedDict
import functools
import threading
import datetime

from util.nn import ModelArchitecture, get_model

try:
    import reverb
except ImportError as ie:
    print(ie, "Reverb is not installed on your system, "
              "meaning prioritized experience replay cannot be used.")
try:
    import wandb

    use_wandb = True
except ImportError as ie:
    print(ie, 'Wandb failed to load; tensorboard will be used instead to log training infos.')
    use_wandb = False

from absl import app
from reinforcement_learning.flags import FLAGS
from reinforcement_learning.environments.pacman.metric import LabelMetric

import PIL

import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import Progbar
from tf_agents.agents.dqn import dqn_agent

import tf_agents
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.metrics import tf_metrics, tf_metric, py_metrics
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.trajectory import experience_to_transitions, Trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver, categorical_q_policy, boltzmann_policy, q_policy, py_tf_eager_policy
import tf_agents.trajectories.time_step as ts
from reinforcement_learning.environments import EnvironmentLoader

default_fc_architecture = ModelArchitecture(hidden_units=(256, 256), activation='relu')


class DQNLearner:
    def __init__(
            self,
            env_name: str,
            env_suite,
            num_iterations: int = int(2e5),
            checkpoint: bool = False,
            initial_collect_steps: int = int(1e4),
            collect_steps_per_iteration: int = 1,
            replay_buffer_capacity: int = int(1e6),
            network_fc_layer_params: ModelArchitecture = default_fc_architecture,
            network_conv_layer_params: Optional[ModelArchitecture] = None,
            gamma: float = 0.99,
            target_update_period: int = 20,
            target_update_tau: Float = 1.0,
            learning_rate: float = 1e-3,
            log_interval: int = 2500,
            num_eval_episodes: int = 30,
            eval_interval: int = int(1e4),
            parallelization: bool = True,
            num_parallel_environments: int = 4,
            batch_size: int = 64,
            debug: bool = False,
            save_directory_location: str = '.',
            prioritized_experience_replay: bool = False,
            priority_exponent: float = 0.6,
            seed: Optional[int] = 42,
            reward_scaling: int = 1,
            epsilon_greedy: Optional[FloatOrReturningFloat] = 0.1,
            boltzmann_temperature: Optional[FloatOrReturningFloat] = None,
            final_exploration_step: Optional[int] = None,
    ):
        self.parallelization = parallelization and not prioritized_experience_replay

        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = batch_size
        if parallelization:
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_environments
            collect_steps_per_iteration = max(1, collect_steps_per_iteration // num_parallel_environments)

        self.env_name = env_name
        self.env_suite = env_suite
        self.num_iterations = num_iterations

        self.initial_collect_steps = initial_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.log_interval = log_interval

        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval

        self.parallelization = parallelization
        self.num_parallel_environments = num_parallel_environments

        self.batch_size = batch_size

        self.prioritized_experience_replay = prioritized_experience_replay

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        if final_exploration_step is not None:
            if epsilon_greedy is not None:
                class EpsilonGreedy:
                    step = self.global_step
                    schedule = tf.optimizers.schedules.PolynomialDecay(
                        1.,
                        decay_steps=final_exploration_step,
                        end_learning_rate=tf.cast(epsilon_greedy, tf.float32))

                    def __call__(self, *args, **kwargs):
                        return self.schedule(self.step)

                epsilon_greedy = EpsilonGreedy()
            if boltzmann_temperature is not None:
                class BoltzmannTemperature:
                    step = self.global_step
                    schedule = tf.optimizers.schedules.PolynomialDecay(
                        tf.cast(boltzmann_temperature, tf.float32),
                        decay_steps=final_exploration_step, end_learning_rate=1e-1)

                    def __call__(self, *args, **kwargs):
                        return self.schedule(self.step)

                boltzmann_temperature = BoltzmannTemperature()

        env_loader = EnvironmentLoader(env_suite, seed=seed)

        if parallelization:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name)] * num_parallel_environments))
            self.tf_env.reset()
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            # self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = env_loader.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)

        self.observation_spec = self.tf_env.observation_spec()
        self.action_spec = self.tf_env.action_spec()

        # initialize WAE-MDP
        print("Observation Spec")
        print("================")
        observation_spec = tf.nest.flatten(self.tf_env.observation_spec())
        for _spec in observation_spec:
            print(f'> {_spec.name}: shape={_spec.shape}, {_spec.dtype}, bounds=[{_spec.minimum}, {_spec.maximum}]')

        if type(self.tf_env.observation_spec()) in [dict, OrderedDict]:
            preprocessing_layers = {
                name: tf.keras.Sequential([
                                              tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
                                          ] + ([
                                                   tf.keras.layers.Conv2D(
                                                       filters=network_conv_layer_params.filters[i],
                                                       kernel_size=network_conv_layer_params.kernel_size[i],
                                                       strides=network_conv_layer_params.strides[i],
                                                       activation=network_conv_layer_params.activation,
                                                   ) for i in range(len(network_conv_layer_params.filters))
                                               ] if len(spec.shape) == 3 and network_conv_layer_params.is_cnn else []) +
                                          [tf.keras.layers.Flatten(), ]
                                          ) for name, spec in self.tf_env.observation_spec().items()
            }
        else:
            preprocessing_layers = [
                tf.keras.Sequential([
                                        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
                                    ] + ([tf.keras.layers.Conv2D(
                    filters=network_conv_layer_params.filters[i],
                    kernel_size=network_conv_layer_params.kernel_size[i],
                    strides=network_conv_layer_params.strides[i],
                    activation=network_conv_layer_params.activation,
                ) for i in range(len(network_conv_layer_params.filters))
                                         ] if len(spec.shape) == 3 and network_conv_layer_params.is_cnn else []) + [
                                        tf.keras.layers.Flatten(),
                                    ]) for spec in observation_spec]

        self.q_network = q_network.QNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
            fc_layer_params=network_fc_layer_params.hidden_units,
            activation_fn=network_fc_layer_params.activation, )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.tf_agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=self.q_network,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step,
            target_update_period=target_update_period,
            target_update_tau=target_update_tau,
            gamma=gamma,
            reward_scale_factor=reward_scaling,
            epsilon_greedy=epsilon_greedy,
            boltzmann_temperature=boltzmann_temperature,
            # emit_log_probability=True
        )

        self.tf_agent.initialize()

        # define the policy from the learning agent
        self.collect_policy = self.tf_agent.collect_policy

        self.max_priority = tf.Variable(0., trainable=False, name='max_priority', dtype=tf.float64)
        if self.prioritized_experience_replay:
            if checkpoint:
                checkpoint_path = os.path.join(save_directory_location, 'saves', env_name, 'reverb')
                reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(checkpoint_path)
            else:
                reverb_checkpointer = None

            table_name = 'prioritized_replay_buffer'
            table = reverb.Table(
                table_name,
                max_size=replay_buffer_capacity,
                sampler=reverb.selectors.Prioritized(priority_exponent=priority_exponent),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1))

            if checkpoint:
                reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer)
            else:
                reverb_server = reverb.Server([table])

            self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                data_spec=self.tf_agent.collect_data_spec,
                sequence_length=2,
                table_name=table_name,
                local_server=reverb_server)

            _add_trajectory = reverb_utils.ReverbAddTrajectoryObserver(
                py_client=self.replay_buffer.py_client,
                table_name=table_name,
                sequence_length=2,
                stride_length=1,
                priority=self.max_priority)

            self.num_episodes = py_metrics.NumberOfEpisodes()
            self.env_steps = py_metrics.EnvironmentSteps()
            self.avg_return = py_metrics.AverageReturnMetric(buffer_size=50)
            observers = [self.num_episodes, self.env_steps, self.avg_return, _add_trajectory]

            self.driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=observers,
                max_steps=collect_steps_per_iteration)
            self.initial_collect_driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=[_add_trajectory],
                max_steps=initial_collect_steps)

        else:
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.tf_agent.collect_data_spec,
                batch_size=self.tf_env.batch_size,
                max_length=replay_buffer_capacity)

            self.num_episodes = tf_metrics.NumberOfEpisodes()
            self.env_steps = tf_metrics.EnvironmentSteps()
            self.avg_return = tf_metrics.AverageReturnMetric(batch_size=self.tf_env.batch_size, buffer_size=50)
            # specific to the pacman grid environment
            self.label_metric = LabelMetric(batch_size=self.tf_env.batch_size)

            observers = [self.num_episodes, self.env_steps] if not parallelization else []
            observers += [self.avg_return, self.replay_buffer.add_batch,
                          self.label_metric]
            # A driver executes the agent's exploration loop and allows the observers to collect exploration information
            self.driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
            self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env,
                self.collect_policy,
                observers=[self.replay_buffer.add_batch],
                num_steps=initial_collect_steps)

        # Dataset generates trajectories with shape [Bx2x...]
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=num_parallel_environments,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

        if checkpoint:
            self.checkpoint_dir = os.path.join(save_directory_location, 'saves', env_name, 'dqn_training_checkpoint')
            self.train_checkpointer = common.Checkpointer(
                ckpt_dir=self.checkpoint_dir,
                max_to_keep=1,
                agent=self.tf_agent,
                policy=self.collect_policy,
                replay_buffer=self.replay_buffer,
                global_step=self.global_step
            )
        else:
            self.checkpoint_dir = None
            self.train_checkpointer = None

        self.policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'dqn_policy')
        self.policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(
            save_directory_location, 'logs', 'gradient_tape', env_name, 'dqn_agent_training', current_time)
        if use_wandb:
            self.train_summary_writer = None
        else:
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = os.path.join(save_directory_location, 'saves', env_name)

        if checkpoint and os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            self.global_step = tf.compat.v1.train.get_global_step()
            print("Checkpoint loaded! global_step={}".format(self.global_step.numpy()))
        if not os.path.exists(self.policy_dir):
            os.makedirs(self.policy_dir)

    def train_and_eval(self, display_progressbar: bool = True, display_interval: float = 0.1):

        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        if not self.prioritized_experience_replay:
            self.driver.run = common.function(self.driver.run)

        metrics = [
            'eval_avg_returns',
            'avg_eval_episode_length',
            'replay_buffer_frames',
            'training_avg_returns',
            #
            'game_over', 'goal_reached'
        ]
        if not self.parallelization:
            metrics += ['num_episodes', 'env_steps']

        train_loss = 0.

        # load the checkpoint

        def update_progress_bar(num_steps=1):
            if display_progressbar:
                log_values = [
                    ('loss', train_loss),
                    ('replay_buffer_frames', self.replay_buffer.num_frames()),
                    ('training_avg_returns', self.avg_return.result()),
                    ('goal_reached', self.label_metric.result()['goal']),
                    ('game_over', self.label_metric.result()['game_over']),
                ]
                if not self.parallelization:
                    log_values += [
                        ('num_episodes', self.num_episodes.result()),
                        ('env_steps', self.env_steps.result())
                    ]
                progressbar.add(num_steps, log_values)

        if display_progressbar:
            progressbar = Progbar(target=self.num_iterations, interval=display_interval, stateful_metrics=metrics)
        else:
            progressbar = None

        env = self.tf_env if not self.prioritized_experience_replay else self.py_env

        if tf.math.less(self.replay_buffer.num_frames(), self.initial_collect_steps):
            print("Initialize replay buffer...")
            self.initial_collect_driver.run(env.current_time_step())

        print("Start training...")

        update_progress_bar(self.global_step.numpy())

        for _ in range(self.global_step.numpy(), self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.driver.run(env.current_time_step())

            # Use data from the buffer and update the agent's network.
            # experience = replay_buffer.gather_all()
            experience, info = next(self.iterator)
            if self.prioritized_experience_replay:
                is_weights = tf.cast(
                    tf.stop_gradient(tf.reduce_min(info.probability[:, 0, ...])) / info.probability[:, 0, ...],
                    dtype=tf.float32)
                loss_info = self.tf_agent.train(experience, weights=is_weights)
                train_loss = loss_info.loss

                priorities = tf.cast(tf.abs(loss_info.extra.td_error), tf.float64)
                self.replay_buffer.update_priorities(keys=info.key[:, 0, ...], priorities=priorities)
                if tf.reduce_max(priorities) > self.max_priority:
                    self.max_priority.assign(tf.reduce_max(priorities))
            else:
                loss_info = self.tf_agent.train(experience)
                train_loss = loss_info.loss

            step = self.tf_agent.train_step_counter.numpy()

            update_progress_bar()

            if step % self.log_interval == 0:
                if self.train_checkpointer:
                    self.train_checkpointer.save(self.global_step)
                    if self.prioritized_experience_replay:
                        self.replay_buffer.py_client.checkpoint()
                self.policy_saver.save(self.policy_dir)

                if self.train_summary_writer is not None:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss, step=step)
                        tf.summary.scalar('training average returns', self.avg_return.result(), step=step)
                elif use_wandb:
                    logs = {
                        'loss': train_loss,
                        'training average returns': self.avg_return.result(),
                        'goal reached': self.label_metric.result()['goal'],
                        'game over': self.label_metric.result()['game_over'],
                    }
                    if self.tf_agent._epsilon_greedy is not None:
                        epsilon = self.tf_agent._epsilon_greedy
                        logs['epsilon_greedy'] = epsilon() if callable(epsilon) else epsilon
                    if self.tf_agent._boltzmann_temperature is not None:
                        temperature = self.tf_agent._boltzmann_temperature
                        logs['boltzmann_temperature'] = temperature() if callable(temperature) else temperature
                    wandb.log(logs, step=step)

            if step % self.eval_interval == 0:
                eval_thread = threading.Thread(target=self.eval, args=(step, progressbar), daemon=True, name='eval')
                eval_thread.start()

    def eval(self, step: int = 0, progressbar: Optional = None):
        avg_eval_return = tf_metrics.AverageReturnMetric(buffer_size=50)
        label_metric = LabelMetric(buffer_size=50)
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric(buffer_size=50)
        saved_policy = tf.compat.v2.saved_model.load(self.policy_dir)
        eval_env = tf_py_environment.TFPyEnvironment(self.env_suite.load(self.env_name))
        eval_env.reset()

        dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            saved_policy,
            [avg_eval_return, avg_eval_episode_length, label_metric],
            num_episodes=self.num_eval_episodes
        ).run()

        log_values = [
            ('eval_avg_returns', avg_eval_return.result()),
            ('eval_avg_episode_length', avg_eval_episode_length.result()),
            ('eval_goal_reached', label_metric.result()['goal']),
            ('eval_game_over', label_metric.result()['game_over']),
        ]
        if progressbar is not None:
            progressbar.add(0, log_values)
        else:
            print('Evaluation')
            for key, value in log_values:
                print(key, '=', value.numpy())
        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Average returns', avg_eval_return.result(), step=step)
                tf.summary.scalar('Average episode length', avg_eval_episode_length.result(), step=step)
        if use_wandb:
            wandb.log({
                'eval_avg_return': avg_eval_return.result(),
                'eval_avg_episode_length': avg_eval_episode_length.result(),
                'eval_goal_reached': label_metric.result()['goal'],
                'eval_game_over': label_metric.result()['game_over'],
            }, step=self.global_step.numpy())


def main(params):
    _params = FLAGS.flag_values_dict()
    _params.update(params)
    params = _params
    # set seed
    seed = params['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_seed(params['seed'])

    try:
        import importlib
        for module in params['import']:
            importlib.import_module(module)
    except BaseException as err:
        serr = str(err)
        print("Error to load module: " + serr)
        return -1

    if use_wandb:
        wandb.init(project='dqn', entity=params['wandb_entity'])

        if params['rerun']:
            run_id = params['rerun']

            api = wandb.Api(timeout=120)
            run = api.run(f'{run_id}')
            run_params = run.config
            for p in params.keys():
                if p in run_params.keys():
                    if p not in [
                        'checkpoint', 'display_progressbar', 'save_dir',
                        'log_videos', 'rerun', 'wandb_entity', 'checkpoint', 'seed'
                    ] and not (p == 'env_name' and params[p] is not None):
                        params[p] = run_params[p]

            params['save_dir'] = os.path.join(params['save_dir'], f'run_{run.id}', f'seed_{params["seed"]}')

            if 'num_units_per_hidden_layer' in run_params.keys() and 'num_hidden_layers' not in run_params.keys():
                params['network_layers'] = [params['num_units_per_hidden_layer']] * params['num_hidden_layers']

        wandb.config.update(params)

    learner = DQNLearner(
        env_name=params['env_name'],
        env_suite=importlib.import_module('tf_agents.environments.' + params['env_suite']),
        num_iterations=params['steps'],
        num_parallel_environments=params['n_parallel_envs'],
        save_directory_location=params['save_dir'],
        learning_rate=params['policy_learning_rate'],
        network_fc_layer_params=ModelArchitecture(
            hidden_units=params['network_layers'],
            activation=params['activation']),
        network_conv_layer_params=ModelArchitecture(
            filters=params['cnn_filters'],
            kernel_size=params['cnn_kernel_size'],
            strides=params['cnn_strides'],
            padding=params['cnn_padding'],
            activation=params['cnn_layers_activation']),
        batch_size=params['policy_batch_size'],
        parallelization=params['n_parallel_envs'] > 1,
        epsilon_greedy=params['epsilon_greedy'] if not params['boltzmann_temperature'] else None,
        final_exploration_step=params['final_exploration_step'],
        boltzmann_temperature=params['boltzmann_temperature'],
        target_update_period=params['target_update_period'],
        target_update_tau=params['target_update_scale'],
        gamma=params['gamma'],
        collect_steps_per_iteration=params['collect_steps_per_iteration'],
        prioritized_experience_replay=params['prioritized_experience_replay'],
        priority_exponent=params['priority_exponent'],
        seed=params['seed'],
        reward_scaling=params['reward_scaling'],
        log_interval=params['log_interval'],
        checkpoint=params['checkpoint'],
    )
    learner.train_and_eval(display_progressbar=params['display_progressbar'])
    if use_wandb:
        wandb.finish()
    return 0


def app_main(argv):
    del argv
    return main(dict())


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, app_main))
    # app.run(main)
