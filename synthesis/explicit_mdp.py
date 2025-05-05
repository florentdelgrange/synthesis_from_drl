import time
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import graph_mode
from tf_agents.environments import suite_gym
from tf_agents.policies import TFPolicy
import tensorflow_probability as tfp
from tf_agents.typing.types import Float

from synthesis.model_checking import \
    retrieve_epsilon_greedy_distribution, get_p_init
from verification.value_iteration import value_iteration
from wasserstein_mdp import WassersteinMarkovDecisionProcess
from reinforcement_learning.environments.two_level_env import TwoLevelEnv
from reinforcement_learning.environments.two_level_env import Directions
from synthesis.entrance_function import EntranceFunction

try:
    import wandb

    use_wandb = True
except ImportError as ie:
    use_wandb = False

tfd = tfp.distributions


class ExplicitMDP:

    def __init__(
            self,
            latent_models: Dict[Directions, Dict[str, Union[WassersteinMarkovDecisionProcess, TFPolicy]]],
            values: Dict[Directions, Dict[str, tf.Tensor]],
            two_level_env: TwoLevelEnv,
            env_prefix: str,
            entrance_function: Optional[EntranceFunction] = None,
            grid_path: Optional[str] = None,
    ):

        self.values = values
        self.latent_models = latent_models
        self.two_level_env = two_level_env
        self.entrance_function = entrance_function
        self.grid_path = grid_path
        self._q_values = None
        self._env_prefix = env_prefix

        self.time_metrics = dict()
        self.time_metrics['high level model generation'] = time.time()
        self._generate_model()
        self.time_metrics['high level model generation'] = \
            time.time() - self.time_metrics['high level model generation']

    def _generate_model(self):
        # the state space of the grid MDP consists of
        # the product of the rooms and the input directions
        self.n_states = len(self.two_level_env.rooms) * len(Directions)
        # one policy per direction (noop is excluded)
        n_policies = len(Directions) - 1
        # special states
        self.sink = self.n_states
        self.goal = self.n_states + 1
        # initial room, and initial direction
        initial_directions_per_room = self.two_level_env.initial_directions

        succinct_mdp = np.zeros(shape=(self.n_states + 2, n_policies, self.n_states + 2))

        # make sink and goal states absorbing
        for action in range(n_policies):
            succinct_mdp[self.sink, action, self.sink] = 1.
            succinct_mdp[self.goal, action, self.goal] = 1.

        # map each room and each input direction to the linked state in the grid MDP
        high_level_map = np.zeros(shape=(len(self.two_level_env.rooms), len(Directions)), dtype=int)
        inverse_map = np.zeros(shape=(self.n_states, 2), dtype=int)
        for room, _ in enumerate(self.two_level_env.rooms):
            for direction in Directions:
                high_level_map[room, direction] = room * len(Directions) + direction
                inverse_map[room * len(Directions) + direction] = np.array([room, direction])

        # map each MDP state to True when self.goal can be reached when
        # executing target_direction in that state;
        # goal_directions[room, direction] is true iff
        # there is a goal in the input room located in the input direction
        goal_directions = np.array([
            [
                self.two_level_env.get_goal_direction(room) == direction
                for direction in Directions
                if direction != Directions.NOOP
            ] for room in range(len(self.two_level_env.rooms))
        ])

        for state in range(self.n_states):

            room, input_direction = inverse_map[state]

            # detect irrelevant states
            if (
                    input_direction not in self.two_level_env.get_available_directions(room) + [Directions.NOOP]
                    and not (
                    room in initial_directions_per_room.keys()
                    and input_direction in initial_directions_per_room[room])
            ):
                for direction in [_d for _d in Directions if _d != Directions.NOOP]:
                    succinct_mdp[state, direction, self.sink] = 1.
            else:
                for target_direction in [_d for _d in Directions if _d != Directions.NOOP]:

                    if target_direction in self.two_level_env.get_available_directions(room):

                        next_room = self.two_level_env.get_next_room(room, target_direction)

                        if room != next_room:
                            next_input_direction = self.two_level_env.get_entrance_direction(
                                room, next_room, target_direction)
                        else:
                            next_input_direction = Directions.NOOP

                        if goal_directions[room, target_direction]:
                            next_state = self.goal
                        else:
                            next_state = high_level_map[next_room, next_input_direction]

                        if self.entrance_function is not None:
                            p_init = self.entrance_function(input_direction, target_direction, room)
                        else:
                            # use initial distribution of each room instead of the entrance function
                            wae_mdp = self.latent_models[target_direction]['model']
                            policy = self.latent_models[target_direction]['policy']
                            policy_distribution = retrieve_epsilon_greedy_distribution(wae_mdp, policy)
                            with suite_gym.load(
                                    f'{self._env_prefix}HighLevel-v0',
                                    gym_kwargs={'grid_path': self.grid_path}
                            ) as py_env:
                                p_init = get_p_init(
                                    wae_mdp, py_env, policy_distribution,
                                    f'{self._env_prefix}HighLevel-v0')

                        v_success = self.values[target_direction]['succeed']
                        p_success = tf.squeeze(p_init[None] @ v_success[..., None])

                        succinct_mdp[state, target_direction, next_state] = p_success
                        succinct_mdp[state, target_direction, self.sink] = 1. - p_success

                    else:
                        succinct_mdp[state, target_direction, self.sink] = 1.
        self.succinct_mdp = succinct_mdp
        self.map = high_level_map
        self.inverse_map = inverse_map
        self.initial_directions_per_room = initial_directions_per_room

    def render(self, q_values: Optional[Float] = None):
        import matplotlib.pyplot as plt
        import networkx as nx

        assert use_wandb, "Wandb must be enabled to render the MDP."

        num_states, num_actions, _ = self.succinct_mdp.shape
        side = np.floor(np.sqrt(self.two_level_env.n_rooms))
        space_size = 50 * side

        # Create a directed graph for the MDP
        mdp_graph = nx.DiGraph()

        positions = np.array([
            [
                {
                    Directions.NOOP: np.array([1, 1]),
                    Directions.DOWN: np.array([1, 0]),
                    Directions.UP: np.array([1, 2]),
                    Directions.LEFT: np.array([0, 1]),
                    Directions.RIGHT: np.array([2, 1])
                }[direction] + np.array(
                    [(room % side) * side, (room // side) * side]
                ) * np.array([1, -1]) for direction in Directions
            ] for room in range(len(self.two_level_env.rooms))
        ]) * space_size

        # Add nodes for each state
        for state in range(num_states):

            if state not in [self.sink, self.goal]:
                room, input_direction = self.inverse_map[state]

                if (
                        input_direction in self.two_level_env.get_available_directions(room) + [Directions.NOOP]
                        or
                        (room in self.initial_directions_per_room.keys()
                         and input_direction in self.initial_directions_per_room[room])
                ):
                    pos = positions[room, input_direction]

                    input_direction = str(
                        Directions(input_direction)
                    ).replace('Directions.', '')

                    mdp_graph.add_node(state, label=f'{room}, {input_direction}', pos=pos)

            elif state == self.goal:
                for room_number in range(self.two_level_env.n_rooms):
                    direction = self.two_level_env.get_goal_direction(room_number)
                    if direction is not None:
                        goal_position = (positions[room_number, direction] + space_size * np.array([-1, 0]))
                        mdp_graph.add_node(state, label='GOAL', pos=goal_position)
                        break

        # Add edges for transitions
        for state in range(num_states):

            if state not in [self.goal, self.sink]:
                room, input_direction = self.inverse_map[state]
            else:
                room = input_direction = None

            if (state in [self.goal, self.sink]) or (
                    input_direction in self.two_level_env.get_available_directions(room) + [Directions.NOOP]) or (
                    room in self.initial_directions_per_room.keys()
                    and input_direction in self.initial_directions_per_room[room]
            ):
                if q_values is None:
                    for action in range(num_actions):
                        for next_state in range(num_states):
                            probability = self.succinct_mdp[state, action, next_state]
                            if probability > 0 and next_state != self.sink and not (state == next_state == self.goal):
                                action_label = str(Directions(action)).replace('Directions.', '')
                                mdp_graph.add_edge(state, next_state,
                                                   label=f'[{str(action_label)}] {probability:.4g}')
                else:
                    strategy = lambda room, input_direction: self.optimal_strategy(room, input_direction, q_values)
                    action = strategy(room, input_direction)
                    for next_state in range(num_states):
                        probability = self.succinct_mdp[state, action, next_state]
                        if probability > 0 and next_state != self.sink and not (state == next_state == self.goal):
                            action_label = str(Directions(action)).replace('Directions.', '')
                            mdp_graph.add_edge(
                                state, next_state,
                                label=f'Q(., {action_label})={q_values[state, action]:.4g}')

        # Define node positions for layout
        pos = {node: data['pos'] for node, data in mdp_graph.nodes(data=True)}

        # Extract labels for nodes and edges
        node_labels = {node: data['label'] for node, data in mdp_graph.nodes(data=True)}
        success_edge_labels = {
            (u, v): data['label'] for u, v, data in mdp_graph.edges(data=True)
            if (self.goal == v and self.goal != u) or
               (self.goal != u and self.inverse_map[u][0] != self.inverse_map[v][0])
        }
        retry_edge_labels = {
            (u, v): data['label'] for u, v, data in mdp_graph.edges(data=True)
            if self.goal not in [u, v] and self.inverse_map[u][0] == self.inverse_map[v][0]}

        fig, ax = plt.subplots(figsize=(side * 5, side * 5), dpi=300)

        # Draw nodes, edges, and labels
        nx.draw(mdp_graph, pos, with_labels=False, node_size=2000, font_size=1 if q_values is None else 2,
                node_color='lightblue', edge_color='gray' if q_values is None else 'red', width=2, arrowsize=10)
        nx.draw_networkx_labels(mdp_graph, pos, labels=node_labels,
                                font_color='black', font_size=10)
        nx.draw_networkx_edge_labels(mdp_graph, pos, edge_labels=success_edge_labels,
                                     font_color='green', font_size=4 if q_values is None else 3, )
        nx.draw_networkx_edge_labels(mdp_graph, pos, edge_labels=retry_edge_labels,
                                     font_color='black', font_size=4 if q_values is None else 3, )

        # Show the graph
        plt.axis('off')
        wandb.log({'hierarchical mdp': wandb.Image(fig)})

    def get_q_values(self, gamma: float = .99) -> Float:
        reward_map = np.zeros(shape=self.succinct_mdp.shape, dtype=float)
        reward_map[..., self.goal] = 1.
        reward_map[self.goal, ..., self.goal] = 0.

        self.time_metrics['value_iteration'] = time.time()

        q_values = value_iteration(
            latent_state_size=12,  # dummy value; will be ignored since transition/reward matrices are provided
            num_actions=len(Directions) - 1,
            gamma=gamma,
            policy=None,
            epsilon=1e-6,
            episodic_return=False,
            v_init=tf.zeros(self.succinct_mdp.shape[:2], dtype=tf.float32),
            transition_fn=self.succinct_mdp,
            reward_fn=reward_map,
            transition_matrix=tf.cast(self.succinct_mdp, tf.float32),
            reward_matrix=tf.cast(reward_map, tf.float32),
            return_q_values=True)

        self.time_metrics['value_iteration'] = time.time() - self.time_metrics['value_iteration']
        self._q_values = q_values
        return q_values

    def optimal_strategy(
            self,
            room: int,
            input_direction: Directions,
            q_values: Optional[Float] = None,
            softmax: bool = False,
    ) -> int:
        if q_values is None:
            if self._q_values is None:
                q_values = self.get_q_values()
            else:
                q_values = self._q_values
        if None in [room, input_direction]:
            return 0
        state = self.map[room, input_direction]
        available_directions = self.two_level_env.get_available_directions(room)
        mask = np.sum(np.eye(len(Directions) - 1)[available_directions], axis=0)
        if softmax:
            probs = tfd.Categorical(
                logits=q_values[state, ...]
            ).probs_parameter().numpy()
            probs = probs * mask / np.sum(probs * mask, axis=-1)
            return tfd.Categorical(probs=probs).sample().numpy()
        else:
            candidate_directions = (q_values[state, ...] == tf.reduce_max(q_values[state, ...], axis=-1)).numpy()
            available_directions = mask * 2. - 1.
            candidate_directions = np.where(candidate_directions * available_directions == 1.)[0]
            return np.random.choice(candidate_directions)

    def __str__(self):
        color_code = 31
        to_print = ''
        for state in range(self.n_states):
            room, input_direction = self.inverse_map[state]

            if (
                    input_direction in self.two_level_env.get_available_directions(room) + [Directions.NOOP]
                    or
                    (room in self.initial_directions_per_room.keys()
                     and input_direction in self.initial_directions_per_room[room])
            ):

                for target_direction in self.two_level_env.get_available_directions(room):
                    action_selected = f'[{str(Directions(target_direction)).replace("Directions.", "")}] '
                    if self._q_values is not None and self.optimal_strategy(room, input_direction) == target_direction:
                        action_selected = f"\033[{color_code}m{action_selected}\033[0m"
                    to_print += action_selected
                    preamble = f'(room={room} & input_direction='
                    preamble += f'{str(Directions(input_direction)).replace("Directions.", "")})'
                    to_print += preamble + ' -> '

                    next_states = np.where(self.succinct_mdp[state, target_direction, ...] != 0.)[0]

                    for i, next_state in enumerate(next_states):
                        if next_state not in [self.goal, self.sink]:
                            next_room, next_direction = self.inverse_map[next_state]
                            to_print += f'(room={next_room} & input_direction=' \
                                        f'{str(Directions(next_direction)).replace("Directions.", "")}) : ' \
                                        f'{self.succinct_mdp[state, target_direction, next_state]:.6f}'
                        elif next_state == self.sink:
                            to_print += f'SINK : ' \
                                        f'{self.succinct_mdp[state, target_direction, next_state]:.6f}'
                        elif next_state == self.goal:
                            to_print += f'GOAL : ' \
                                        f'{self.succinct_mdp[state, target_direction, next_state]:.6f}'

                        if i + 1 < len(next_states):
                            to_print += ' + '

                    if self._q_values is not None:
                        to_print += f'  |  Q = {self._q_values[state, target_direction]:.6f}'

                    to_print += '\n'
        return to_print
