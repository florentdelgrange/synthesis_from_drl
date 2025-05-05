import sys

from reinforcement_learning.environments.two_level_env import Directions

if __name__ == '__main__':
    from tf_agents.environments import suite_gym, tf_py_environment
    from tf_agents.policies import random_tf_policy
    from tf_agents.drivers import dynamic_episode_driver
    from tf_agents.trajectories import StepType

    from reinforcement_learning.environments.perturbed_env import PerturbedEnvironment

    args = sys.argv[1:]
    if '--human' in args:
        human = True
    else:
        human = False

    env_name = 'PacmanGrid-v0'

    if '--direction' in args:
        direction = {
            'south': Directions.DOWN,
            'down': Directions.DOWN,
            'north': Directions.UP,
            'up': Directions.UP,
            'east': Directions.RIGHT,
            'right': Directions.RIGHT,
            'west': Directions.LEFT,
            'left': Directions.LEFT,
        }[args[args.index('--direction') + 1].lower()]
    else:
        direction = None

    with suite_gym.load(env_name, gym_kwargs={'direction': direction}) as py_env:
        if human:
            done = False
            py_env.reset()
            py_env.render(mode='human')
            while not done:
                action = input()
                action = action[0] if len(action) > 0 else 'X'
                action = {
                    'z': Directions.UP,
                    's': Directions.DOWN,
                    'q': Directions.LEFT,
                    'd': Directions.RIGHT,
                }.get(action[0], Directions.NOOP)
                step = py_env.step(action)
                done = step.step_type == StepType.LAST
                py_env.render(mode='human')
            print("Done!")
        else:
            py_env = PerturbedEnvironment(env=py_env, perturbation=.1)

            if '--tf' in args:
                tf_env = tf_py_environment.TFPyEnvironment(py_env)
                tf_policy = random_tf_policy.RandomTFPolicy(
                    action_spec=tf_env.action_spec(),
                    time_step_spec=tf_env.time_step_spec())

                driver = dynamic_episode_driver.DynamicEpisodeDriver(
                    env=tf_env,
                    policy=tf_policy,
                    observers=[
                        # lambda _: py_env.render(mode='human'),
                    ],
                    num_episodes=int(args[args.index('--episodes') + 1]) if '--episodes' in args else 5,
                ).run()
            else:
                py_env.reset()
                PyDriver(
                    env=py_env,
                    policy=RandomPyPolicy(time_step_spec=py_env.time_step_spec(), action_spec=py_env.action_spec()),
                    observers=[
                        # lambda _: py_env.render(mode='human')
                    ],
                    max_episodes=int(args[args.index('--episodes') + 1]) if '--episodes' in args else 5,
                ).run(time_step=py_env.current_time_step())

            pass
