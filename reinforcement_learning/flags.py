from absl import flags

default_flags = set(flags.FLAGS.flag_values_dict().keys())

# =========================================================
# RL Flags
# =========================================================
flags.DEFINE_string(
    'env_name', help='Name of the environment', default=None,
)
flags.DEFINE_string(
    'env_suite', help='Environment suite', default='suite_gym'
)
flags.DEFINE_integer(
    'n_parallel_envs',
    help='Number of parallel environments',
    default=4
)
flags.DEFINE_integer(
    'env_time_limit',
    default=None,
    help='(Optional) enforce the environment to reset after the input time limit'
)
flags.DEFINE_float(
    'env_perturbation',
    default=0.25,
    help="Probability of the recursive environment perturbation. "
         "If < 1, the environment is recursively perturbed when reset; in that case, the input value corresponds to "
         "the probability of going to the initial state. This enforces the ergodicity of the environment"
)
flags.DEFINE_multi_integer(
    'network_layers',
    help='number of units per fully connected Dense layers',
    default=[128, 128]
)
flags.DEFINE_integer(
    'policy_batch_size',
    help='Batch size for learning the policy',
    default=64
)
flags.DEFINE_float(
    'policy_learning_rate',
    help='learning rate',
    default=3e-4
)
flags.DEFINE_float(
    'policy_adam_epsilon',
    help='Adam optimizer epsilon',
    default=1e-7,
)
flags.DEFINE_integer(
    'collect_steps_per_iteration',
    help='Collect steps per iteration',
    default=None
)
flags.DEFINE_integer(
    'collect_episodes_per_iteration',
    help='Collect episodes per iteration',
    default=None
)
flags.DEFINE_integer(
    'initial_collect_steps',
    help="Number of collect steps to perform in the environment before performing an update",
    default=None
)
flags.DEFINE_integer(
    'initial_collect_episodes',
    help="Number of collect episodes to perform in the environment before performing an update",
    default=None
)
flags.DEFINE_integer(
    'target_update_period',
    help="Period for update of the target networks",
    default=20
)
flags.DEFINE_float(
    'target_update_scale',
    help='Weights scaling for the target network updates. '
         'Set to 1 to perform hard updates and < 1 for soft updates',
    default=1.
)
flags.DEFINE_float(
    'gamma',
    help='discount_factor',
    default=0.99
)
flags.DEFINE_integer(
    'replay_buffer_size',
    help='Replay buffer maximum capacity',
    default=int(1e6)
)
flags.DEFINE_bool(
    'prioritized_experience_replay',
    help="Use priority-based replay buffer (via Deepmind reverb)",
    default=False
)
flags.DEFINE_float(
    'priority_exponent',
    help='priority exponent for computing the probabilities of the samples from the prioritized replay buffer',
    default=0.6
)
flags.DEFINE_float(
    'importance_sampling_exponent',
    help='exponent applied on importance sampling weights, annealed up to 1 along training steps',
    default=0.5
)
flags.DEFINE_integer(
    'n_eval_episodes',
    help='Number of episodes to perform for evaluating the policy',
    default=30
)
flags.DEFINE_integer(
    'n_eval_interval',
    help='Number of steps to perform before evaluating the policy',
    default=int(1e4)
)
flags.DEFINE_float(
    'epsilon_greedy',
    help='Epsilon value for the epsilon greedy based exploration policy',
    default=0.1
)
flags.DEFINE_float(
    'final_exploration_step',
    help='Number of steps over which the initial value of epsilon (or Boltzmann)'
         ' is linearly annealed to its final value.',
    default=None
)
flags.DEFINE_float(
    'boltzmann_temperature',
    help='(Optional) softmax temperature for a Boltzmann exploration policy',
    default=None,
)
flags.DEFINE_float(
    'reward_scaling',
    help='Scale factor for the rewards',
    default=1.
)
flags.DEFINE_float(
    'policy_gradient_clipping',
    help='(Optional) norm length to clip gradients',
    default=None
)
flags.DEFINE_bool(
    'log_grads_and_vars',
    help="Whether to log gradients per variable",
    default=False,
)
flags.DEFINE_multi_integer(
    'cnn_filters',
    help='Number of convolution filters. Only used when the observation space has two dimensions or more',
    default=[16, 32, 32],
)
flags.DEFINE_multi_integer(
    'cnn_kernel_size',
    help='Kernel size of convolution layers. Only used when the observation space has two dimensions or more.',
    default=[4, 2, 3]
)
flags.DEFINE_multi_integer(
    'cnn_strides',
    help='Convolution layers strides. Only used when the observation space has two dimensions or more.',
    default=[2, 1, 1]
)
flags.DEFINE_bool(
    'use_max_pooling',
    help='Whether to use max pooling between convolution layers or not',
    default=False
)
flags.DEFINE_string(
    'cnn_layers_activation',
    help='Activation function used for the CNN layers.',
    default='relu',
)
flags.DEFINE_multi_string(
    'cnn_padding',
    help='Convolution layers padding. Only used when the observation space has two dimensions or more.',
    default=["valid", "valid", "valid"]
)
flags.DEFINE_multi_integer(
    'state_components_concat_units',
    default=[32],
    help='When provided and the state space has multiple components, define the number of units used per state'
         ' component to concatenate them and form a flattened input (after their possible pre-processing).'
)
flags.DEFINE_bool(
    'pre_processing_net_raw_last',
    default=False,
    help='If set, only use the output of the pre_processing_layers as encoder output. In that case, when the state '
         'space has several components, the concatenation of the different flattened pre_processing outputs are'
         ' used as "raw" output of the encoder.'
)

# =========================================================
# HER flags
# =========================================================

flags.DEFINE_bool(
    'her',
    help='use hindsight experience replay. Currently only compatible with uniform replay buffers.',
    default=False
)
flags.DEFINE_string(
    'her_sampling_strategy',
    help='Hindsight experience replay sampling strategy (either future or final)',
    default='future'
)
flags.DEFINE_integer(
    'her_n_future_samples',
    help='Number of future goal to sample',
    default=4
)
flags.DEFINE_float(
    'her_time_step_cost',
    help='Cost of each non-final time step to write in the HER',
    default=1.
)
flags.DEFINE_float(
    'her_state_penalty_multiplier',
    help='HER state penalty multiplier',
    default=1.
)
flags.DEFINE_float(
    'her_goal_reward_multiplier',
    help='HER goal reward multiplier',
    default=0.
)
flags.DEFINE_integer(
    'her_reward_horizon',
    help='HER reward horizon',
    default=100
)
flags.DEFINE_integer(
    'her_cycles',
    help='Number of HER cycles per epoch',
    default=50,
)
flags.DEFINE_integer(
    'her_optimization_steps',
    help='Number of optimization steps to perform per HER cycle',
    default=40
)
flags.DEFINE_integer(
    'her_epochs',
    help='Number of HER epochs. '
         'If not provided, the number of epochs is computed according to --steps.',
    default=None
)

# =========================================================
# WAE-MDP Flags
# =========================================================
flags.DEFINE_string(
    'activation',
    help='Activation function for the fully connected Dense hidden layers of the WAE-MDP model',
    default='relu'
)
flags.DEFINE_integer(
    'wae_batch_size',
    help='Batch size for the WAE',
    default=128,
)
flags.DEFINE_integer(
    'n_wae_critic',
    help="number of Wasserstein critic (discriminators) updates before performing a full WAE-MDP update",
    default=5,
)
flags.DEFINE_integer(
    'n_wae_updates',
    help='number of WAE-MDP updates before performing a policy update',
    default=1,
)
flags.DEFINE_integer(
    'latent_state_size',
    help='Number of bits to use to represent the state space',
    default=10,
)
flags.DEFINE_float(
    'state_encoder_temperature',
    help='Temperature of the state encoder (encoding each original states to relaxed Bernoulli)',
    default=2. / 3.,
    lower_bound=0.,
    upper_bound=1.,
)
flags.DEFINE_float(
    'state_prior_temperature',
    help='Temperature of the transition function stationary distribution (for producing relaxed Bernoullis)',
    default=1. / 2.,
    lower_bound=0.,
    upper_bound=1.,
)
flags.DEFINE_float(
    'transition_regularizer_scale_factor',
    help='Scale factor for the WAE-MDP transition regularizer',
    default=60.
)
flags.DEFINE_float(
    'steady_state_regularizer_scale_factor',
    help='Scale factor for the WAE-MDP steady-state regularizer',
    default=60.
)
flags.DEFINE_float(
    'gradient_penalty_scale_factor',
    help='Scale factor for penalizing the gradients of the WAE-MDP regularizers',
    default=10.,
)
flags.DEFINE_float(
    'wae_minimizer_learning_rate',
    help='Learning rate for the Wasserstein Autoencoder part (for the min operation)',
    default=3e-4
)
flags.DEFINE_float(
    'wae_maximizer_learning_rate',
    help='Learning rate for the WAE-MDP regularizers (for the max operation)',
    default=3e-4
)
flags.DEFINE_float(
    required=False,
    name='encoder_learning_rate',
    default=None,
    help='If provided, use a separate optimizer for the encoder built with the input learning rate'
)
flags.DEFINE_integer(
    'wae_eval_steps',
    help='Number of steps to perform to evaluate the WAE loss (with discrete latent spaces, i.e., via the zero limit '
         'temperature)',
    default=int(1e4)
)
flags.DEFINE_multi_string(
    'state_cost_fn',
    help='Cost function for reconstructing the states. '
         'If the state space have multiple (N) components, then either one (the same for all) or N '
         'cost functions should be provided.',
    default=['l22']
)
flags.DEFINE_string(
    'reward_cost_fn',
    help='Cost function for reconstructing the rewards',
    default='l2'
)
flags.DEFINE_multi_float(
    'state_cost_weights',
    help='weight the loss function for each state component',
    default=1.
)
flags.DEFINE_float(
    'reward_cost_weight',
    help='weight of the reward function for each state component',
    default=1.
)
flags.DEFINE_multi_integer(
    'conditional_labeling',
    help='indices (i, j) of the latent space vector z indicating the conditional label z[i:j] (e.g., in the case where'
         ' the label encodes the goal of the agent)',
    default=None
)
flags.DEFINE_integer(
    'conditional_units',
    default=128,
    help='Number of units to use to combine conditional components (e.g., the state and the goal for a goal '
         'conditioned policy). '
)
flags.DEFINE_bool(
    'use_batch_norm',
    help='whether to use batch norm or not',
    default=False,
)
flags.DEFINE_bool(
    'categorical',
    help='Whether to use distributional Q-learning or not',
    default=False,
)
flags.DEFINE_integer(
    'num_atoms',
    help='Number of atoms for the categorical distributional Q-learning',
    default=51,
)
flags.DEFINE_float(
    'min_q_value',
    help='Minimum value for the categorical distributional Q-learning',
    default=-10.,
)
flags.DEFINE_float(
    'max_q_value',
    help='Maximum value for the categorical distributional Q-learning',
    default=10.,
)
flags.DEFINE_integer(
    'n_step_update',
    help='Number of steps to perform before updating the policy (to further compute the n-step return)',
    default=1
)
flags.DEFINE_bool(
    name='concatenate_losses',
    default=False,
    help='Whether to concatenate the RL TD error with the WAE-MDP loss. '
         'If set, the optimization is performed on the sum of the two losses as follows:'
         'loss = (1 - alpha) * td_error + alpha * wae_loss'
)
flags.DEFINE_float(
    name='concatenate_alpha',
    default=0.01,
    help='Weight of the WAE-MDP loss in the concatenated loss'
)
flags.DEFINE_float(
    name='wae_mdp_clip_by_global_norm',
    default=None,
    help='(Optional) If provided, clip the gradients of the WAE-MDP loss by the provided global norm'
)
flags.DEFINE_bool(
    name='train_on_discrete_states',
    default=True,
    help='Whether to train the policy with discrete states or relaxed states while interacting with the environment'
)
flags.DEFINE_bool(
    'steady_state_softclip',
    help='Whether to softclip the output of the steady-state network or not',
    default=False
)
flags.DEFINE_bool(
    'straight_through',
    help='Whether to use the straight-through estimator for latent states',
    default=False
)
flags.DEFINE_string(
    'softclip_fn',
    help='Softclip function to use for bounding outputs of NNs',
    default='softclip'
)
flags.DEFINE_bool(
    'use_total_variation',
    help='Whether to use the total variation instead of the Wasserstein distance',
    default=False
)
flags.DEFINE_bool(
    'anneal_temperature',
    help='Whether to anneal the temperature of the encoder and the transition/steady-state networks',
    default=False
)
flags.DEFINE_integer(
    'final_temperature_step',
    help='Number of steps over which the initial temperature is linearly annealed to its final value',
    default=None,
)
# =========================================================
# Utils
# =========================================================
flags.DEFINE_integer(
    'steps',
    help='Total number of iterations. Ignored if her_epoch is provided',
    default=int(2e5)
)
flags.DEFINE_multi_string(
    'import',
    help='list of modules to additionally import',
    default=[]
)
flags.DEFINE_integer(
    'seed', help='set seed', default=42
)
flags.DEFINE_string(
    'save_dir', help='save/checkpoint directory location', default='.'
)
flags.DEFINE_integer(
    'log_interval',
    help='Number of global steps before logging',
    default=200
)
flags.DEFINE_bool(
    'display_progressbar',
    help='Whether to display a progressbar or not during training',
    default=True,
)
flags.DEFINE_string(
    'log_name',
    help='log name',
    default=None
)
flags.DEFINE_bool(
    'checkpoint',
    help='Whether to checkpoint the model during its training.',
    default=True
)
flags.DEFINE_string(
    'wandb_entity',
    help='Weight and Biases entity',
    default=None
)
flags.DEFINE_multi_string(
    name='tags',
    default=[],
    help='Tags to log to wandb.'
)
flags.DEFINE_bool(
    'log_videos',
    help='Whether to produce videos of the policy during the evaluation or not.',
    default=False
)
flags.DEFINE_string(
    'rerun',
    help='If set, re-run an instance of the algorithm via the provided Weight and Biases run ID',
    default=None
)
flags.DEFINE_float(
    'time_limit',
    help='(Optional) time limit for running the current algorithm (in sec)',
    default=None)
flags.DEFINE_bool(
    'keep_last_model',
    help='Whether to save the last model or not',
    default=True
)
flags.DEFINE_bool(
    name='eval_thread',
    default=True,
    help='Whether to run the evaluation in a separate thread or not'
)
FLAGS = flags.FLAGS
