import logging
import os
import sys

import tf_agents
from optuna._callbacks import RetryFailedTrialCallback

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

from reinforcement_learning.wae_dqn import main

from typing import Optional, Tuple, Dict

import numpy as np
import optuna


def optimize_hyperparameters(study_name, env_name, optimize_trial, storage=None, n_trials=100):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if storage is None:
        if not os.path.exists(os.path.join('studies')):
            os.makedirs(os.path.join('studies'))
        storage = f'sqlite:///studies/{study_name}.db'

    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={
            'connect_args': {'timeout': sqlite_timeout},
        },
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        study_name=study_name if env_name is None else env_name,
        storage=storage,
        load_if_exists=True,
        # maximize the mean score, minimize the std, minimize the latent space size
        # directions=['maximize', 'minimize', 'minimize'])
        direction='maximize')  # , 'minimize', 'minimize'])

    return study.optimize(optimize_trial, n_trials=n_trials)


def suggest_hyperparameters(trial, parameters):
    hyperparams = dict()

    def _suggest(_param, _config, _suggestion_type):

        def _str_categorical():
            cat_params = {str(x): x for x in _config['choices']}
            suggestion = trial.suggest_categorical(_param, cat_params.keys())
            return cat_params[suggestion]

        hyperparams[_param] = {
            'str_categorical': _str_categorical,
            'categorical': lambda: trial.suggest_categorical(_param, _config['choices']),
            'int': lambda: trial.suggest_int(
                _param, _config['min'], _config['max']),
            'constant': lambda: _config['value']
        }[_suggestion_type]()

    for param, config in parameters['hyperparameters'].items():
        if param != 'conditionals':
            assert 'type' in config, f'Please provide a suggestion type for parameter {param}'
            suggestion_type = config['type']
            _suggest(param, config, suggestion_type)

    def _set_key(_dict, key, value):
        _dict[key] = value
        return 0

    def _str_replace(str_: str, dict_: Dict[str, str]):
        for word, replacement in dict_.items():
            str_ = str_.replace(word, str(replacement))
        return str_

    conditionals = []
    for conditional in parameters['hyperparameters']['conditionals']:
        class ConditionalStatement:
            _conditional_parameters = conditional['vars']
            _cond_statement = conditional['cond']
            _if_cond = conditional['if_cond']
            _else = conditional['else']

            def cond_fn(self):
                return eval(_str_replace(
                    self._cond_statement,
                    {conditional_param: hyperparams[conditional_param]
                     for conditional_param in self._conditional_parameters}))

            def true_fn(self):
                return [_suggest(_param, _config, _config['type'])
                        for _param, _config in self._if_cond.items()]

            def false_fn(self):
                [_suggest(_param, _config, _config['type'])
                 for _param, _config in self._else.items()]

        conditional = ConditionalStatement()
        conditionals.append((conditional.cond_fn, conditional.true_fn, conditional.false_fn))

    for cond, true_fn, false_fn in conditionals + [
        # custom conditional suggestions
        (lambda: hyperparams.get('pre_processing_net_raw_last', False) and 'state_component_mask' in parameters,
         lambda: _set_key(
             hyperparams,
             'state_components_concat_units',
             _raw_last_units(
                 trial,
                 parameters['label_size'],
                 parameters['state_component_mask'],
                 hyperparams['latent_state_size'])),
         lambda: None)
    ]:
        if cond():
            true_fn()
        else:
            false_fn()

    print("Suggested hyperparameters")
    for key in hyperparams.keys():
        if key != "specs":
            print(f"{key}={hyperparams[key]}")

    return hyperparams


# ================================
# Custom suggestion functions
# ================================
def _raw_last_units(trial, label_size: int, _mask: Tuple[bool, ...], latent_state_size: int):
    remaining_unmasked_components = len(np.where(_mask)[0])
    last_index = np.where(_mask)[0][-1]
    state_components_concat_units = []
    available_units = latent_state_size - label_size
    for i, _take_unit in enumerate(_mask):
        if _take_unit and i < last_index:
            remaining_unmasked_components -= 1
            state_components_concat_units.append(
                trial.suggest_int(
                    f'state_component_{i:d}',
                    1, available_units - remaining_unmasked_components))
            available_units -= state_components_concat_units[-1]
        elif i == last_index:
            state_components_concat_units.append(available_units)
        else:
            state_components_concat_units.append(0)
    return state_components_concat_units


# ================================
# Custom parameter renaming
# ================================
def parameters_renaming(parameters: Dict):
    """
    Rename or add entries in the dict containing the parameters of the model to be optimized by optuna
    Args:
        parameters: dict containing the parameters of the model to optimize
    """
    if 'cnn_config' in parameters:
        parameters['cnn_filters'] = parameters['cnn_config']['filters']
        parameters['cnn_kernel_size'] = parameters['cnn_config']['kernel']
        parameters['cnn_strides'] = parameters['cnn_config'].get('strides', [1] * len(parameters['cnn_filters']))
        parameters['cnn_padding'] = parameters['cnn_config'].get('padding', ['valid'] * len(parameters['cnn_filters']))
    if 'num_units_per_hidden_layer' in parameters and 'num_hidden_layers' in parameters:
        parameters['network_layers'] = tuple(
            parameters['num_units_per_hidden_layer'] for _ in range(parameters['num_hidden_layers']))


def search(
        parameters: Dict,
        study_name='study',
        n_trials=100,
):
    def optimize_trial(trial: optuna.multi_objective.trial):
        hyperparameters = suggest_hyperparameters(trial, parameters)
        parameters_renaming(hyperparameters)
        parameters['log_name'] = f'trial_number={trial.number:d}'
        parameters.pop('hyperparameters')

        eval_window = main({**parameters, **hyperparameters})

        for i, score in enumerate(eval_window):
            trial.report(score, step=(i + 1) * parameters['n_eval_interval'])

        return eval_window.mean()

    if type(parameters['env_name']) == str:
        env_name = parameters['env_name']
    else:
        parameters['env_name'] = np.random.choice(parameters['env_name'])
        env_name = None

    return optimize_hyperparameters(study_name, env_name, optimize_trial, n_trials=n_trials)


if __name__ == '__main__':
    import yaml
    import argparse
    import absl.flags as flags

    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument('--params', type=str, nargs=1,
                        help='path of a yaml file containing the parameters of this study')
    args = parser.parse_args()
    yaml_file = args.params[0]

    flags.DEFINE_string(
        name='params',
        default=None,
        help='path of a yaml file containing the parameters of this study'
    )
    flags = flags.FLAGS.flag_values_dict()

    assert yaml_file is not None, 'The yaml file containing the parameters has to be provided (via --params).'

    with open(yaml_file, 'r') as file:
        parameters = yaml.safe_load(file)

    tf_agents.system.multiprocessing.handle_main(
        lambda _: search(parameters, parameters.get('study_name', 'wae_dqn_study'),
                         parameters['n_trials']))
