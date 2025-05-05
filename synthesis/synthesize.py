import functools
import os, sys
import random
from typing import Dict, Collection, Optional, Union, Callable

import gym
import numpy as np
import tf_agents
from absl import app
from absl import flags
from prettytable import PrettyTable
import tensorflow as tf
import tensorflow.keras as tfk
from tf_agents.environments import tf_py_environment, suite_gym
from tf_agents.policies import TFPolicy
from tf_agents.trajectories import StepType
from tf_agents.typing.types import Int
from gym.wrappers.time_limit import TimeLimit

try:
    import wandb

    use_wandb = True
except ImportError as ie:
    use_wandb = False

path = os.path.dirname(os.path.abspath("__file__"))
root_path = os.path.join(path, '..')
sys.path.insert(0, root_path)

from synthesis.entrance_function import EntranceFunction
from synthesis.io import load_policies
from reinforcement_learning.environments.two_level_env import Directions, TwoLevelEnv
from synthesis.explicit_mdp import ExplicitMDP
from synthesis.model_checking import compute_latent_mdp_values, \
    save_values, load_values, get_pac_bounds
from reinforcement_learning import labeling_functions
from util.io.dataset_generator import ergodic_batched_labeling_function
from wasserstein_mdp import WassersteinMarkovDecisionProcess
from util.io.video import VideoEmbeddingObserverNumpy

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

DEBUG: int = 0

default_flags = set(flags.FLAGS.flag_values_dict().keys())

flags.DEFINE_string(
    name='grid_path',
    default=None,
    help='Path to grid file defining the environment.'
)
flags.DEFINE_bool(
    name='use_frequency_estimator',
    default=False,
    help='Whether to use the frequency estimator.'
)
flags.DEFINE_integer(
    name='dataset_size',
    default=1024,
    help='Size of the dataset to use for estimating the entrance function.'
)
flags.DEFINE_string(
    name='dataset_path',
    default='dataset',
    help='Path to directory where the dataset should be saved / is stored.'
)
flags.DEFINE_bool(
    name='save_dataset',
    default=False,
    help='Whether to save the dataset.'
)
flags.DEFINE_bool(
    name='load_dataset',
    default=False,
    help='Whether to load the dataset.'
)
flags.DEFINE_bool(
    name='save_autoregressive_model',
    default=False,
    help='Whether to save the autoregressive model.'
)
flags.DEFINE_bool(
    name='load_autoregressive_model',
    default=False,
    help='Whether to load the autoregressive model.'
)
flags.DEFINE_string(
    name='autoregressive_model_path',
    default='models',
    help='Path to directory where the autoregressive model should be saved / is stored.'
)
flags.DEFINE_multi_integer(
    name='hidden_units',
    default=[64, 64],
    help='number of units per layer for the autoregressive network modeling the entrance function.'
)
flags.DEFINE_integer(
    name='conditional_units',
    default=64,
    help='number of (latent) units to use to combine neural network layers.'
)
flags.DEFINE_string(
    name='activation',
    default='relu',
    help='activation function to use for the autoregressive network modeling the entrance function.'
)
flags.DEFINE_float(
    name='learning_rate',
    default=1e-3,
    help='learning rate of the autoregressive network modeling the entrance function.',
)
flags.DEFINE_integer(
    name='batch_size',
    default=32,
    help='batch size used for training the autoregressive network modeling the entrance function.',
)
flags.DEFINE_bool(
    name='batch_norm',
    default=False,
    help='Whether to use batch normalization for the autoregressive network modeling the entrance function.',
)
flags.DEFINE_bool(
    name='layer_norm',
    default=False,
    help='Whether to use layer normalization for the autoregressive network modeling the entrance function.',
)
flags.DEFINE_integer(
    name='epochs',
    default=100,
    help='number of epochs for training the autoregressive network modeling the entrance function.',
)
flags.DEFINE_string(
    name='values_path',
    default='latent_values',
    help='Path to directory where the latent values are stored.'
)
flags.DEFINE_integer(
    name='max_low_level_steps',
    default=200,
    help='Maximum number of steps to perform before resetting the high-level environment.'
)
flags.DEFINE_integer(
    name='n_eval_episodes',
    default=30,
    help='Number of episodes to run the high-level environment for evaluation.'
)
flags.DEFINE_bool(
    name='estimate_entrance_function',
    default=True,
    help='Whether to estimate the entrance function.'
)
flags.DEFINE_integer(
    name='agent_lives',
    default=3,
    help="Number of agent's lives (i.e., trials within each room before hard resetting the environment)."
         "Not used if the environment does not use lives to count agent's trials "
         "(examples using lives: Pacman; not using lives: Doom)."
)
flags.DEFINE_bool(
    name='render',
    default=False,
    help="Whether to render the environment during the final evaluation."
)
flags.DEFINE_integer(
    name='debug',
    default=0,
    help="Whether to run in debug mode. The higher the value, the more verbose the output."
         "With debug >= 2, breakpoints might be triggered."
)
flags.DEFINE_float(
    name='discount',
    default=.99,
    help="Discount factor."
)
flags.DEFINE_bool(
    name='pac_bounds',
    default=False,
    help='Whether to PAC estimate the transition and reward losses.'
)
flags.DEFINE_float(
    name='delta',
    default=.1,
    help='Confidence parameter for PAC bounds.'
)
flags.DEFINE_float(
    name='epsilon',
    default=.05,
    help='Accuracy parameter for PAC bounds.'
)
flags.DEFINE_integer(
    name='average_episode_length',
    default=100,
    help='Average episode length. Used to determine the sampling procedure for estimating the PAC bounds.'
)
flags.DEFINE_integer(
    name='seed',
    default=42,
    help='Random seed.'
)
flags.DEFINE_bool(
    name='log_video',
    default=False,
    help='Whether to record a video of the evaluation (log the result on wandb).'
)
flags.DEFINE_bool(
    name='exploit_latent_values',
    default=False,
    help='Whether to exploit the latent value function during the exploration performed '
         'for learning the entrance function.'
)
flags.DEFINE_string(
    name='latent_policies_json_path',
    default='latent_policies.json',
    help='Path to JSON pointing to the location of the latent models.'
)
flags.DEFINE_multi_string(
    name='tags',
    default=[],
    help='Tags to log to wandb.'
)
flags.DEFINE_bool(
    name='batch_mode',
    default=None,
    help='Whether to use batch mode for computing the latent MDP values.', )
flags.DEFINE_string(
    name='env_prefix',
    default='PacmanGrid',
    help="Prefix of the environment's name."
)
flags.DEFINE_bool(
    name='use_relaxed_states',
    default=False,
    help='Whether to use relaxed states when running the policies in the environment.'
)


def generate_time_table(log_dict, columns: Collection[str]) -> PrettyTable:
    table = PrettyTable()

    # Define columns
    table.field_names = columns

    # Add data to the table
    for step, time in log_dict.items():
        table.add_row([step, time])

    return table


def get_env_name_with_prefix(prefix: str) -> str:
    """
    Get the name of a gym environment that has the input name as a prefix.

    Args:
        prefix (str): The prefix to search for in the environment names.

    Returns:
        str: The name of the first environment that matches the prefix.
    """
    envs = gym.envs.registry.all()
    for env in envs:
        if env.id.startswith(prefix):
            return env.id
    return None


def simulate_high_level_environment(
        latent_models: Dict[Directions, Dict[str, Union[WassersteinMarkovDecisionProcess, TFPolicy]]],
        env_prefix: str,
        grid_path: Optional[str] = None,
        dataset: Optional[Dict[str, tf.Tensor]] = None,
        dataset_size: Optional[int] = None,
        max_low_level_steps: int = 200,
        n_episodes: Optional[int] = None,
        high_level_policy: Callable[[Int, Int], Int] = None,
        discount: float = .99,
        agent_lives: int = 3,
        render: bool = False,
        log_video: bool = False,
        use_relaxed_states=False,
        *args, **kwargs,
) -> Dict[str, Union[TwoLevelEnv, float]]:
    assert (all(arg is not None for arg in [dataset, dataset_size])) != (n_episodes is not None), \
        "Either dataset and dataset_size or n_episodes must be specified."

    if grid_path is None:
        grid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grid_easyEval.txt')

    max_latent_size = max([
        latent_models[direction]['model'].latent_state_size
        for direction in Directions
        if direction != Directions.NOOP])

    progbar = tfk.utils.Progbar(
        target=dataset_size if dataset is not None else n_episodes,
        stateful_metrics=['episodes'])

    with suite_gym.load(
            f'{env_prefix}HighLevel-v0',
            gym_kwargs={'grid_path': grid_path, 'lives': agent_lives, },
            gym_env_wrappers=(lambda env: TimeLimit(env, max_episode_steps=max_low_level_steps),)
    ) as py_env:
        two_level_env = py_env.gym
        _env_name = get_env_name_with_prefix(env_prefix)
        labeling_fn = ergodic_batched_labeling_function(labeling_functions[_env_name])
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        tf_env.reset()
        time_out = game_over = False
        episodes = goal_reached = 0
        values = []

        assert not log_video or use_wandb, "wandb must be initialized to log videos."

        if log_video:
            video_observer = VideoEmbeddingObserverNumpy(py_env)
        else:
            video_observer = None

        while True:

            continue_ = True
            value = 1.
            two_level_env.hard_reset()

            while continue_ and (
                    (dataset is not None and len(dataset['input_direction']) < dataset_size) or
                    (n_episodes is not None and episodes < n_episodes)
            ):

                if (time_out or game_over) and DEBUG:
                    print(
                        '\n' if DEBUG > 1 else '',
                        f'[DEBUG] [to; fail] = {time_out, game_over}')
                elif two_level_env.has_been_hard_reset and DEBUG:
                    print(
                        '\n' if DEBUG > 1 else '',
                        '[DEBUG] *hard reset*')

                input_direction = int(two_level_env.input_direction)

                if dataset is not None:
                    dataset['input_direction'].append(input_direction)
                    dataset['room_number'].append(two_level_env.current_room)

                if high_level_policy:
                    direction = high_level_policy(two_level_env.current_room, input_direction)
                else:
                    direction = np.random.choice(two_level_env.available_directions)
                two_level_env.set_low_level_objective(direction)
                
                traversal_counter=0
                while two_level_env.between_two_rooms:
                    py_env.step(two_level_env.default_action)
                    if log_video:
                        video_observer()
                    traversal_counter += 1
                    if traversal_counter > 500:
                        break
                if traversal_counter > 500:
                    if dataset is not None:
                        dataset['input_direction'].pop()
                        dataset['room_number'].pop()
                    print('[Warning] the agent is stuck between two rooms | '
                            f'dest room=({two_level_env.current_room},'
                            f'{str(Directions(input_direction))})'
                            f' => [{str(Directions(direction))}]')
                    break

                if dataset is not None:
                    dataset['output_direction'].append(int(direction))

                if DEBUG:
                    print(
                        '\n' if DEBUG > 1 else '',
                        f'[DEBUG] room=({two_level_env.current_room}, {str(Directions(input_direction))})'
                        f' => [{str(Directions(direction))}]', )
                    if DEBUG > 1:
                        input()

                state_embedding_fn = lambda state: latent_models[direction]['model'].state_embedding_function(
                    state, labeling_fn(state), use_discrete_states=not use_relaxed_states, dtype=tf.float32)
                latent_state = state_embedding_fn(tf_env.current_time_step().observation)[0]
                if use_relaxed_states:
                    latent_state = tf.round(latent_state)
                latent_size = tf.shape(latent_state)[-1]
                latent_state = tf.pad(latent_state, [[0, max_latent_size - latent_size]])

                if dataset is not None:
                    dataset['latent_state'].append(latent_state)

                policy = latent_models[direction]['policy']
                
                info = py_env.get_info()
                if info is None:
                    info = dict()
                goal_reached += int(info.get('goal_reached', False))
                done = py_env.current_time_step().step_type == StepType.LAST
                while not done and (
                        (dataset is not None and len(dataset['input_direction']) < dataset_size) or
                        (n_episodes is not None and episodes < n_episodes)
                ):
                    if render:
                        py_env.render(mode='human')
                    if log_video:
                        video_observer()
                    latent_state = state_embedding_fn(tf_env.current_time_step().observation)
                    latent_ts = tf_env.current_time_step()._replace(
                        observation=latent_state)
                    action = policy.distribution(latent_ts).action.sample()
                    tf_env.step(action)

                    done = py_env.current_time_step().step_type == StepType.LAST
                    info = {**py_env.get_info()}

                    if 'TimeLimit.truncated' in info:
                        time_out = info['TimeLimit.truncated']
                        del info['TimeLimit.truncated']
                        info['time_out'] = time_out
                    else:
                        info['time_out'] = time_out = False

                    goal_reached += int(info['goal_reached'])
                    game_over = info.get('game_over', False)
                    value *= discount

                if render:
                    py_env.render(mode='human')
                if log_video:
                    video_observer()

                py_env.reset()
                continue_ = not two_level_env.has_been_hard_reset
                if two_level_env.has_been_hard_reset:
                    episodes += 1
                    values.append(value * int(info.get('goal_reached', False)))
                    if DEBUG:
                        print(
                            '\n' if DEBUG > 1 else '',
                            f'[DEBUG] episode {episodes} ended with value {value * int(info["goal_reached"])}')
                        if DEBUG > 1:
                            input()

                # if not DEBUG:
                #     progbar.update(
                #         episodes if dataset is None else len(dataset['input_direction']),
                #         [(key, float(value)) for key, value in info.items()] +
                #         [#("agent's lives", two_level_env.lives),
                #             ("episodes", episodes)])

            if (
                    (dataset is not None and dataset_size <= len(dataset['input_direction'])) or
                    (n_episodes is not None and episodes >= n_episodes)
            ):
                break

        if log_video:
            wandb.log({'video': wandb.Video(np.moveaxis(video_observer.data, -1, 1), fps=12, format="mp4")})

        return {
            'grid': two_level_env,
            'average_goal_reached': goal_reached / max(episodes, 1),
            'value': np.array(values).mean(),
        }


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_seed(seed)


def generate_high_level_model(
        latent_models: Dict[Directions, Dict[str, Union[WassersteinMarkovDecisionProcess, TFPolicy]]],
        values: Dict[Directions, Dict[str, tf.Tensor]],
        two_level_env: TwoLevelEnv,
        env_prefix: str,
        grid_path: Optional[str] = None,
        discount: float = .99,
        entrance_function: Optional[EntranceFunction] = None,
):
    explicit_mdp = ExplicitMDP(
        values=values,
        latent_models=latent_models,
        env_prefix=env_prefix,
        grid_path=grid_path,
        two_level_env=two_level_env,
        entrance_function=entrance_function, )
    explicit_mdp.render()
    q_values = explicit_mdp.get_q_values(gamma=discount)
    return {
        'explicit_mdp': explicit_mdp,
        'q_values': q_values,
    }


def main(argv):
    del argv
    FLAGS = flags.FLAGS
    params = {key: value for key, value in FLAGS.flag_values_dict().items() if key not in default_flags}
    global DEBUG
    DEBUG = params['debug']
    set_seed(params['seed'])

    if use_wandb:
        run = wandb.init(project="synthesis")
        run.tags += tuple(params['tags'])
        wandb.config.update(params)

    latent_models = load_policies(path=root_path, json_path=os.path.join(path, params['latent_policies_json_path']))

    with suite_gym.load(
            get_env_name_with_prefix(f'{params["env_prefix"]}HighLevel'),
            max_episode_steps=params['max_low_level_steps'],
            gym_kwargs={'grid_path': params['grid_path']}
    ) as py_env:
        py_env.reset()
        two_level_env = py_env.gym
        if use_wandb:
            try:
                env_image = wandb.Image(two_level_env.render(mode='rgb_array'))
                image = wandb.Image(env_image, caption="Environment")
                wandb.log({'env_image': image})
            except Exception as e:
                print(f"Could not log environment image: {e}")

    time_metrics = dict()

    # load/compute the values of the latent models/policies
    load_path = os.path.join(path, params['values_path'], f'discount={params["discount"]:.2g}')
    if not os.path.exists(load_path):
        values, _time_metrics = compute_latent_mdp_values(
            latent_models=latent_models,
            discount=params['discount'],
            batch_mode=params['batch_mode'])
        time_metrics = {**time_metrics, **_time_metrics}
        save_values(values, discount=params['discount'], path=load_path)
    else:
        values = load_values(discount=params['discount'], path=load_path)

    def environment_simulator(
            dataset: Optional[Dict[str, tf.Tensor]],
            _,
    ):
        if params['exploit_latent_values']:
            high_level_models = generate_high_level_model(
                latent_models=latent_models,
                values=values,
                env_prefix=params['env_prefix'],
                grid_path=params['grid_path'],
                discount=params['discount'],
                two_level_env=two_level_env
            )

            def _high_level_exploration_policy(room, input_direction):
                return high_level_models['explicit_mdp'].optimal_strategy(
                    room=room,
                    input_direction=input_direction,
                    q_values=high_level_models['q_values'],
                    softmax=True)
        else:
            _high_level_exploration_policy = None

        _params = {**params}
        _params['log_video'] = False

        return simulate_high_level_environment(
            latent_models=latent_models,
            dataset=dataset,
            high_level_policy=_high_level_exploration_policy,
            **_params)

    if params['estimate_entrance_function']:

        entrance_fn = EntranceFunction(
            latent_models=latent_models,
            environment_simulator=environment_simulator,
            save_model=params['save_autoregressive_model'],
            load_model=params['load_autoregressive_model'],
            model_path=params['autoregressive_model_path'],
            **params)
        if not params['load_autoregressive_model']:
            time_metrics = {**entrance_fn.time_metrics}
            two_level_env = entrance_fn.two_level_env
    else:
        entrance_fn = None

    high_level_models = generate_high_level_model(
        latent_models=latent_models,
        values=values,
        grid_path=params['grid_path'],
        two_level_env=two_level_env,
        discount=params['discount'],
        entrance_function=entrance_fn,
        env_prefix=params['env_prefix'],
    )
    explicit_mdp = high_level_models['explicit_mdp']
    explicit_mdp.render()
    q_values = high_level_models['q_values']
    explicit_mdp.render(q_values)
    if DEBUG:
        print(explicit_mdp)
    time_metrics = {**time_metrics, **explicit_mdp.time_metrics}

    value_metrics = dict()
    for initial_room, initial_directions in two_level_env.initial_directions.items():
        for initial_direction in initial_directions:
            v_init = q_values[explicit_mdp.map[initial_room, initial_direction],
            explicit_mdp.optimal_strategy(initial_room, initial_direction, q_values)]
            if 'latent initial value' not in value_metrics:
                value_metrics['latent initial value'] = [v_init]
            else:
                value_metrics['latent initial value'].append(v_init)
    value_metrics['latent initial value'] = tf.reduce_mean(value_metrics['latent initial value']).numpy()

    # PAC bounds

    if params['pac_bounds']:
        print('Computing PAC bounds')
        pac_metrics, _time_metrics = get_pac_bounds(
            latent_models,
            environment_prefix_name=params['env_prefix'],
            **params, )
        value_metrics = {**value_metrics, **pac_metrics}
        time_metrics = {**time_metrics, **_time_metrics}

    print('[High-level strategy evaluation] simulating the grid')
    params['dataset_size'] = None

    simulation_values = simulate_high_level_environment(
        latent_models=latent_models,
        n_episodes=params['n_eval_episodes'],
        high_level_policy=lambda room, input_direction: explicit_mdp.optimal_strategy(
            room, input_direction, q_values),
        **params
    )
    value_metrics['goal reached'] = simulation_values['average_goal_reached']
    value_metrics['estimated initial value'] = simulation_values['value']

    if use_wandb:
        time_table = wandb.Table(
            data=[[key, value] for key, value in time_metrics.items()],
            columns=['step', 'time (s)'])
        value_table = wandb.Table(
            data=[[key, value] for key, value in value_metrics.items()],
            columns=['metric', 'value'])
        wandb.log({
            'time_metrics': time_table,
            'value_metrics': value_table,
            **value_metrics
        })
    print(generate_time_table(time_metrics, ["Step", "Time (s)"]))
    print(generate_time_table(value_metrics, ["Metric", "Value"]))


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, main))
