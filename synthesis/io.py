import json
import os
from typing import Dict, Union

from tf_agents.policies import TFPolicy

import wasserstein_mdp
from wasserstein_mdp import WassersteinMarkovDecisionProcess
from policies.one_hot_categorical import OneHotTFPolicyWrapper
from policies.saved_policy import SavedTFPolicy
from reinforcement_learning.environments.two_level_env import Directions


def load_policies(
        path: str,
        json_path: str
) -> Dict[Directions, Dict[Directions, Dict[str, Union[WassersteinMarkovDecisionProcess, TFPolicy]]]]:
    paths = json.load(open(json_path))

    latent_models = dict()

    for direction_str in list(paths.keys()):
        direction = {
            'Directions.UP': Directions.UP,
            'Directions.DOWN': Directions.DOWN,
            'Directions.LEFT': Directions.LEFT,
            'Directions.RIGHT': Directions.RIGHT,
        }[direction_str]
        paths[direction] = paths[direction_str]
        del paths[direction_str]

    for direction in [direction for direction in Directions if direction != Directions.NOOP]:
        wae_model_path = os.path.join(path, paths[direction]['wae_model'])
        policy_path = os.path.join(path, paths[direction]['policy'])
        wae_mdp = wasserstein_mdp.load(wae_model_path)
        policy = SavedTFPolicy(policy_path)
        wae_mdp.external_latent_policy = OneHotTFPolicyWrapper(
            policy,
            time_step_spec=policy.time_step_spec,
            action_spec=policy.action_spec)
        latent_models[direction] = {'policy': policy, 'model': wae_mdp}

    return latent_models

