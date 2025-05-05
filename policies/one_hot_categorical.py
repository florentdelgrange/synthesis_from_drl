import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tf_agents.policies import tf_policy
import tf_agents.trajectories.time_step as ts

from tf_agents.trajectories import policy_step
from tf_agents.typing import types


class OneHotTFPolicyWrapper(tf_policy.TFPolicy):
    """
    Categorical policy wrapper; changes Categorical to OneHotCategorical in tf.float32 if to_one_hot is set,
    and OneHotCategorical to Categorical otherwise.
    """

    def __init__(self,
                 categorical_policy: tf_policy.TFPolicy,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 use_logits_info: bool = False,
                 to_one_hot: bool = True):
        super().__init__(time_step_spec, action_spec)
        self._policy = categorical_policy
        self._use_logits_info = use_logits_info
        self._to_one_hot = to_one_hot

    def _distribution(
            self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec
    ) -> policy_step.PolicyStep:
        _step = self._policy.distribution(time_step, policy_state)
        if self._to_one_hot:
            Distribution = tfd.OneHotCategorical
        else:
            # from one hot
            Distribution = tfd.Categorical

        if self._use_logits_info:
            assert hasattr(_step.info, 'log_probability'), \
                "The policy does not emit log_probabilities"
            return policy_step.PolicyStep(
                Distribution(logits=_step.info.log_probability, dtype=self.action_spec.dtype), (), ())
        elif type(_step.action) in [tfd.Categorical, tfd.OneHotCategorical]:
            logits = _step.action.logits_parameter()
            return policy_step.PolicyStep(
                Distribution(logits=logits, dtype=self.action_spec.dtype), (), ())
        elif type(_step.action) is tfd.Deterministic:
            loc = _step.action.loc
            return policy_step.PolicyStep(
                Distribution(
                    probs=tf.one_hot(loc, depth=self.action_spec.maximum),
                    dtype=self.action_spec.dtype),
                (), ())
        elif hasattr(_step.info, 'log_probability'):
            return policy_step.PolicyStep(
                Distribution(logits=_step.info.log_probability, dtype=self.action_spec.dtype), (), ())
        else:
            return NotImplemented
