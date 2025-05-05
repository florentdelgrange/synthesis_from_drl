import os.path
import time
from typing import Dict, Callable, Tuple, Optional, Union, Any, Collection

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp

import wasserstein_mdp
from reinforcement_learning.environments.pacman.grid_env import PacmanEnvEval
from reinforcement_learning.environments.two_level_env import Directions
from verification import binary_latent_space

tfd = tfp.distributions
tfb = tfp.bijectors

try:
    import wandb

    use_wandb = True
    from wandb.keras import WandbMetricsLogger
except ImportError as ie:
    use_wandb = False


class EntranceFunction:
    def __init__(
            self,
            latent_models: Dict[Directions, Dict[str, wasserstein_mdp.WassersteinMarkovDecisionProcess]],
            environment_simulator: Callable[[Dict[str, Collection], int], Any],
            use_frequency_estimator: bool = True,
            grid_path: Optional[str] = None,
            dataset_size: int = 1024,
            hidden_units: Tuple[int, ...] = (128, 128),
            conditional_units: int = 64,
            activation_fn: Union[str, Callable] = 'relu',
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            save_dataset: bool = False,
            load_dataset: bool = False,
            dataset_path: Optional[str] = None,
            save_model: bool = False,
            load_model: bool = False,
            model_path: Optional[str] = None,
            batch_norm: bool = False,
            layer_norm: bool = False,
            *args, **kwargs
    ):
        self._grid = None
        self._dataset = None
        self._min_class_weight = None
        self._max_class_weight = None
        self.latent_models = latent_models
        self.latent_space = {
            direction: binary_latent_space(
                latent_models[direction]['model'].latent_state_size
            ) for direction in Directions if direction != Directions.NOOP
        }
        self._use_frequency_estimator = use_frequency_estimator
        self.grid_path = grid_path
        self._autoregressive_model = None
        self.model = None

        if load_model:
            assert model_path is not None, "You must specify a model path"
            self._autoregressive_model = tf.saved_model.load(model_path)
            print(f"Model loaded from {model_path}")
            self.model = self._autoregressive_model
            self._use_frequency_estimator = False
        else:
            if dataset_path and load_dataset:
                self.load_dataset(dataset_path)
                self.two_level_env = environment_simulator(self.dataset, 0)['grid']
            self.time_metrics = self.learn_entrance_function(
                environment_simulator=environment_simulator,
                dataset_size=dataset_size,
                hidden_units=hidden_units,
                conditional_units=conditional_units,
                activation_fn=activation_fn,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                batch_norm=batch_norm,
                layer_norm=layer_norm,)
            if dataset_path and save_dataset:
                self.save_dataset(dataset_path)
            if save_model:
                assert model_path is not None, "You must specify a model path"
                if use_wandb:
                    model_path = os.path.join(model_path, wandb.run.name)
                tf.saved_model.save(self._autoregressive_model, model_path)
                print(f"Model saved at {model_path}")

    def __call__(
            self,
            input_direction: Directions,
            target_direction: Directions,
            room: int,
            *args,
            **kwargs
    ):
        """
        Returns a vector containing the entrance probabilities of each latent state
        """
        latent_state_space = self.latent_space[target_direction]
        n_latent_states = tf.shape(latent_state_space)[0]

        if self._use_frequency_estimator:
            p_init = self.model.probs_parameter()[
                         input_direction, target_direction, room
                     ][:n_latent_states]

        else:
            input_ = tf.repeat([input_direction], n_latent_states)
            output_ = tf.repeat([target_direction], n_latent_states)
            room_ = tf.repeat([room], n_latent_states)

            p_init = self.model.prob(
                input_direction=input_,
                output_direction=output_,
                room_number=room_,
                latent_state=latent_state_space)

        return p_init

    def learn_entrance_function(
            self,
            environment_simulator: Callable[[Dict[str, Collection], int], Any],
            dataset_size: int = 1024,
            hidden_units: Tuple[int, ...] = (128, 128),
            conditional_units: int = 64,
            activation_fn: Union[str, Callable] = 'relu',
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            batch_norm: bool = False,
            layer_norm: bool = False,
    ) -> Dict[str, float]:

        time_metrics = dict()
        if self._dataset is None:
            time_metrics['dataset_generation'] = time.time()
            self.generate_np_dataset(
                simulate_environment=environment_simulator,
                dataset_size=dataset_size, )
            time_metrics['dataset_generation'] = time.time() - time_metrics['dataset_generation']

            time_metrics['dataset_weights'] = time.time()
            self.update_dataset_weights()
            time_metrics['dataset_weights'] = time.time() - time_metrics['dataset_weights']

        time_metrics['autoregressive_model'] = time.time()
        self.train_autoregressive_model(
            self.latent_models,
            hidden_units=hidden_units,
            conditional_units=conditional_units,
            activation_fn=activation_fn,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            batch_norm=batch_norm,
            layer_norm=layer_norm,)
        time_metrics['autoregressive_model'] = time.time() - time_metrics['autoregressive_model']

        if self._use_frequency_estimator:
            time_metrics['frequency_estimator'] = time.time()
            self.model = FrequencyEstimator(
                dataset=self.dataset,
                n_rooms=self.two_level_env.n_rooms,
                latent_models=lambda direction: self.latent_models[direction]['model'],
                backup_model=self._autoregressive_model)
            time_metrics['frequency_estimator'] = time.time() - time_metrics['frequency_estimator']
        else:
            self.model = self._autoregressive_model

        return time_metrics

    def generate_np_dataset(
            self,
            simulate_environment: Callable[[Dict[str, Collection], int], Dict],
            dataset_size: int = 1024,
    ):
        dataset = {
            'input_direction': [],
            'output_direction': [],
            'room_number': [],
            'latent_state': []
        }

        print("[Entrance function estimation] simulating the grid")

        self.two_level_env = simulate_environment(dataset, dataset_size)['grid']
        self.dataset = self._dataset_to_tensor(dataset)

    @staticmethod
    def _dataset_to_tensor(dataset: Dict[str, Union[list, np.ndarray, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        dataset['input_direction'] = tf.constant(dataset['input_direction'])[..., None]
        dataset['output_direction'] = tf.constant(dataset['output_direction'])[..., None]
        dataset['room_number'] = tf.constant(dataset['room_number'])[..., None]
        dataset['latent_state'] = tf.stack(dataset['latent_state'])
        return dataset

    def update_dataset_weights(self):
        n_rooms = len(self.two_level_env.rooms)
        n_classes = n_rooms * len(Directions) * (len(Directions) - 1)
        class_mapping = np.zeros(shape=(n_rooms, len(Directions), (len(Directions) - 1)), dtype=int)
        invert_class_mapping = np.zeros(shape=(n_classes, 3), dtype=int)
        initial_directions_per_room = self.two_level_env.initial_directions

        normalize: bool = False

        counter = 0
        for room in range(n_rooms):
            for input_direction in Directions:
                for target_direction in [d for d in Directions if d != Directions.NOOP]:
                    invert_class_mapping[counter] = (room, input_direction, target_direction)
                    class_mapping[room, input_direction, target_direction] = counter
                    counter += 1

        labels = tf.constant([
            tf.squeeze(class_mapping[room, input_direction, target_direction]).numpy()
            for input_direction, target_direction, room in zip(
                self.dataset['input_direction'],
                self.dataset['output_direction'],
                self.dataset['room_number'])
        ])

        def calculate_class_frequencies(y_train):
            #  unique_classes, class_counts = tf.unique(y_train)
            #  return dict(zip(unique_classes.numpy(), class_counts.numpy()))
            y_train_np = y_train.numpy()
            return dict(zip(*np.unique(y_train_np, return_counts=True)))

        # Calculate class frequencies
        class_frequencies = calculate_class_frequencies(labels)

        # Clean the dataset (detect and remove errors)
        for key, value in class_frequencies.items():
            room, input_direction, direction = invert_class_mapping[key]
            if (Directions(input_direction) not in self.two_level_env.get_available_directions(room) + [Directions.NOOP]
            ) and not (
                    room in initial_directions_per_room.keys() and
                    Directions(input_direction) in initial_directions_per_room[room]
            ):
                class_frequencies[key] = 0.
            if direction not in self.two_level_env.get_available_directions(room) + [Directions.NOOP]:
                class_frequencies[key] = 0.

        # Compute inverse class frequencies as class weights
        total_samples = len(labels)

        class_weights = {
            key: total_samples / (len(class_frequencies) * value) if value != 0. else value
            for key, value in class_frequencies.items()
        }

        # Normalize class weights
        if normalize:
            class_weights = {
                key: value / np.sum(list(class_weights.values()))
                for key, value in class_weights.items()
            }

        self._max_class_weight, self._min_class_weight = max(class_weights.values()), min(class_weights.values())

        self.dataset['weight'] = []

        for room_number, input_direction, output_direction in zip(
                self.dataset['room_number'].numpy(),
                self.dataset['input_direction'].numpy(),
                self.dataset['output_direction'].numpy()
        ):
            self.dataset['weight'].append(
                class_weights[
                    class_mapping[room_number[0], input_direction[0], output_direction[0]]
                ])

        self.dataset['weight'] = tf.stack(self.dataset['weight'])[..., None]

        weight_values = {
            key: (invert_class_mapping[key][0],
                  Directions(invert_class_mapping[key][1]),
                  Directions(invert_class_mapping[key][2]),
                  value)
            for key, value in class_weights.items()
        }.values()

        if use_wandb:
            weight_table = wandb.Table(
                columns=['room', 'input_direction', 'output_direction', 'weight'],
                data=[(value[0], str(value[1]), str(value[2]), value[3])
                      for value in weight_values])
            wandb.log({'weight_table': weight_table})
        else:
            _ = [print(f'room {value[0]}, {str(value[1])} -> {str(value[2])},'
                       f' weight={value[3]:.6f}')
                 for value in weight_values]

    @property
    def two_level_env(self) -> PacmanEnvEval:
        assert self._grid is not None, "You must call generate_np_dataset first"
        return self._grid

    @two_level_env.setter
    def two_level_env(self, grid):
        self._grid = grid

    def save_dataset(self, save_path='dataset'):
        dataset_ = tf.data.Dataset.from_tensor_slices(self.dataset)
        tf.data.experimental.save(dataset_, save_path)

    def load_dataset(self, load_path='dataset'):
        dataset = tf.data.experimental.load(load_path)
        _dataset = {}

        for data in dataset:
            for key, value in data.items():
                if key not in _dataset:
                    _dataset[key] = []
                _dataset[key].append(value)

        for key, value in _dataset.items():
            _dataset[key] = tf.stack(value)

        if self._dataset is None:
            self.dataset = _dataset
        else:
            for key in _dataset.keys():
                if key in self.dataset:
                    self.dataset[key] = tf.concat([self.dataset[key], _dataset[key]], axis=0)

    def train_autoregressive_model(
            self,
            latent_models: Dict[Directions, Dict[str, wasserstein_mdp.WassersteinMarkovDecisionProcess]],
            dataset: Optional[Dict[str, tf.Tensor]] = None,
            hidden_units: Tuple[int, ...] = (128, 128),
            conditional_units: int = 64,
            activation_fn: Union[str, Callable] = 'relu',
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            batch_norm: bool = False,
            layer_norm: bool = False,
            *args, **kwargs
    ):
        if dataset is None:
            dataset = self.dataset
        dataset_size = len(dataset['latent_state'])

        if self._autoregressive_model is None:
            model = AutoregressiveEntranceFunction(
                n_rooms=self.two_level_env.n_rooms,
                latent_models=lambda direction: latent_models[direction]['model'],
                hidden_units=tuple(hidden_units),
                conditional_units=(conditional_units,),
                activation=activation_fn,
                sample_weight=False,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
            )
        else:
            model = self._autoregressive_model

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=lambda _, log_prob: -log_prob)

        model.fit(
            x=dataset,
            y=tf.zeros(shape=(dataset_size,)),
            batch_size=batch_size,
            sample_weight=dataset['weight'],
            epochs=epochs,
            steps_per_epoch=dataset_size // batch_size,
            shuffle=True,
            verbose=True,
            callbacks=[WandbMetricsLogger()] if use_wandb else None,
        )

        self._autoregressive_model = model

    @property
    def dataset(self):
        assert self._dataset is not None, "You must call generate_np_dataset first"
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def use_frequency_estimator(self):
        return self._use_frequency_estimator

    def __str__(self):
        return self.entrance_function.__name__


class AutoregressiveEntranceFunction(tfk.Model):

    def __init__(
            self,
            n_rooms: int,
            latent_models: Callable[[Directions], wasserstein_mdp.WassersteinMarkovDecisionProcess],
            hidden_units: Tuple[int, ...] = (128, 128),
            conditional_units: Tuple[int, ...] = (64, 64),
            activation: str = 'relu',
            batch_norm: bool = False,
            layer_norm: bool = False,
            hadamard_product: bool = False,
            sample_weight: bool = False,
            *args,
            **kwargs
    ):
        hidden_units = list(hidden_units)

        max_latent_size = max([
            latent_models(direction).latent_state_size
            for direction in Directions
            if direction != Directions.NOOP])
        self._n = max_latent_size

        room = tfk.Input(shape=(1,), name='room_number')
        room_one_hot = tfkl.CategoryEncoding(
            num_tokens=n_rooms,
            name='room_one_hot',
        )(room)
        room_embedding = 2. * room_one_hot - 1.
        for i, units in enumerate(conditional_units):
            if i != 0 and batch_norm:
                room_embedding = tfkl.BatchNormalization()(room_embedding)
            if i != 0 and layer_norm:
                room_embedding = tfkl.LayerNormalization()(room_embedding)
            room_embedding = tfkl.Dense(
                units=units,
                activation='tanh',
                name=f'room_embedding_layer_{i:d}'
            )(room_embedding)

        input_direction = tfk.Input(shape=(1,), name='input_direction')
        input_direction_one_hot = tfkl.CategoryEncoding(
            num_tokens=len(Directions),
            name='input_direction_one_hot',
        )(input_direction)
        input_direction_embedding = 2. * input_direction_one_hot - 1.
        for i, units in enumerate(conditional_units):
            if i != 0 and batch_norm:
                input_direction_embedding = tfkl.BatchNormalization()(input_direction_embedding)
            if i != 0 and layer_norm:
                input_direction_embedding = tfkl.LayerNormalization()(input_direction_embedding)
            input_direction_embedding = tfkl.Dense(
                units=units,
                activation='tanh',
                name=f'input_direction_embedding_layer_{i:d}'
            )(input_direction_embedding)

        if hadamard_product:
            conditional_embedding = tfkl.Multiply(name='conditional_embedding')(
                [room_embedding, input_direction_embedding])
        else:
            conditional_embedding = tfkl.Concatenate(name='conditional_embedding')(
                [room_embedding, input_direction_embedding])

        if batch_norm:
            conditional_embedding = tfkl.BatchNormalization()(conditional_embedding)
        if layer_norm:
            conditional_embedding = tfkl.LayerNormalization()(conditional_embedding)

        output_direction = tfk.Input(shape=(1,), name='output_direction')
        output_direction_one_hot = tfkl.CategoryEncoding(
            num_tokens=len(Directions) - 1,
            name='output_direction_one_hot',
        )(output_direction)

        latent_state = tfk.Input(shape=(max_latent_size,), name='latent_state')

        if sample_weight:
            sample_weight_ = tfk.Input(shape=(1,), name='weight')
        else:
            sample_weight_ = None
        self._sample_weight = sample_weight

        directions_log_probs = [None] * 4

        for direction in [d for d in Directions if d != Directions.NOOP]:
            event_shape = (latent_models(direction).latent_state_size,)
            latent_size = event_shape[0]

            if hadamard_product:
                _n = 1
            else:
                _n = 2

            made = tfb.AutoregressiveNetwork(
                params=1,
                event_shape=event_shape,
                conditional=True,
                conditional_event_shape=(conditional_units[-1] * _n,),
                hidden_units=hidden_units,
                activation=activation,
                name=f"autoregressive_net_{str(direction)}")

            distribution = tfd.Autoregressive(
                lambda x: tfd.Independent(
                    tfd.Bernoulli(
                        logits=tf.unstack(
                            made(x, conditional_input=conditional_embedding),
                            axis=-1)[0]),
                    reinterpreted_batch_ndims=1),
                sample0=tf.zeros(event_shape))

            log_prob = distribution.log_prob(latent_state[..., :latent_size])
            directions_log_probs[direction] = log_prob

        directions_log_probs = tf.stack(directions_log_probs, axis=-1)
        log_prob = tfkl.Multiply(name="log_prob")(
            [directions_log_probs, output_direction_one_hot])

        log_prob = tfkl.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1), name='output_lambda_layer'
        )(log_prob)

        if sample_weight:
            reweight = tfkl.Lambda(lambda x: x / tf.reduce_sum(x, axis=0))(sample_weight_)
            log_prob = tfkl.Multiply()([reweight, log_prob])

        super(AutoregressiveEntranceFunction, self).__init__(
            inputs=[
                       latent_state,
                       input_direction,
                       output_direction,
                       room
                   ] + ([sample_weight_] if sample_weight else []),
            outputs=log_prob,
            *args,
            **kwargs)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, ], dtype=tf.int32),
        tf.TensorSpec(shape=[None, ], dtype=tf.int32),
        tf.TensorSpec(shape=[None, ], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        # Add any additional input signatures
    ])
    def prob(self, input_direction, output_direction, room_number, latent_state, *args, **kwargs):
        """
        Gives the probability of latent_state, given the input and output directions, as well
        as a room_number in which the agent is located.
        """
        latent_size = tf.shape(latent_state)[-1]
        latent_state = tf.pad(latent_state, [[0, 0], [0, self._n - latent_size]])
        if self._sample_weight:
            log_probs = self({
                'latent_state': latent_state,
                'input_direction': input_direction,
                'output_direction': output_direction,
                'room_number': room_number,
                'weight': tf.ones_like(room_number)})
        else:
            log_probs = self({
                'latent_state': latent_state,
                'input_direction': input_direction,
                'output_direction': output_direction,
                'room_number': room_number})
        return tf.exp(log_probs)


class FrequencyEstimator:

    def __init__(
            self,
            dataset,
            n_rooms: int,
            latent_models: Callable[[Directions], wasserstein_mdp.WassersteinMarkovDecisionProcess],
            backup_model: AutoregressiveEntranceFunction,
    ):
        max_latent_size = max([
            latent_models(direction).latent_state_size
            for direction in Directions
            if direction != Directions.NOOP])
        self._n = max_latent_size

        state_frequency = np.zeros(
            shape=(
                len(Directions),
                len(Directions) - 1,
                n_rooms,
                2 ** self._n
            ), dtype=int)
        counter = np.zeros(
            shape=(len(Directions), len(Directions) - 1, n_rooms),
            dtype=int)
        latent_states = tf.cast(dataset['latent_state'], tf.int32)
        latent_states = tf.reduce_sum(
            latent_states * 2 ** tf.range(tf.shape(latent_states)[-1]),
            axis=-1
        ).numpy()

        for latent_state, input_direction, output_direction, room_number in zip(
                latent_states,
                dataset['input_direction'].numpy(),
                dataset['output_direction'].numpy(),
                dataset['room_number'].numpy(),

        ):
            state_frequency[input_direction, output_direction, room_number, latent_state] += 1
            counter[input_direction, output_direction, room_number] += 1

        self._probs_parameter = state_frequency / counter[..., None]
        self._model_probs = np.zeros_like(self._probs_parameter)

        for input_direction in Directions:
            for output_direction in [d for d in Directions if d != Directions.NOOP]:
                for room_number in range(n_rooms):
                    latent_size = latent_models(output_direction).latent_state_size
                    input_ = tf.repeat([input_direction], 2 ** latent_size)
                    output_ = tf.repeat([output_direction], 2 ** latent_size)
                    room_ = tf.repeat([room_number], 2 ** latent_size)
                    latent_state_space = binary_latent_space(
                        latent_models(output_direction).latent_state_size,
                        dtype=tf.int32)
                    probs = backup_model.prob(
                        input_direction=input_,
                        output_direction=output_,
                        room_number=room_,
                        latent_state=latent_state_space)
                    probs = tf.pad(probs, [[0, 2 ** self._n - 2 ** latent_size]])
                    self._model_probs[
                        input_direction, output_direction, room_number, ...
                    ] = probs.numpy()

    def probs_parameter(self):
        return tf.cast(
            tf.where(
                tf.math.is_nan(self._probs_parameter),
                self._model_probs,
                self._probs_parameter, ),
            tf.float32)
