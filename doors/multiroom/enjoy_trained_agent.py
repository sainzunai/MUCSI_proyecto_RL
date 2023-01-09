import sys
import gymnasium
sys.modules["gym"] = gymnasium

import gymnasium as gym

from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from minigrid.wrappers import FlatObsWrapper


from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env_str = 'MiniGrid-MultiRoom-N4-S5-v0'
algo = "ppo"

model_str = 'MiniGrid-MultiRoom-N6-v0'


def create_env():
    env = gym.make(env_str, max_steps=100)
    env = FlatObsWrapper(env)
    return env

env = make_vec_env(create_env, n_envs=8)

if algo == "ppo":
    model = PPO.load("models/{}_{}".format(algo, model_str), env=env)
else:
    if algo == "dqn":
        model = DQN.load("models/{}_{}".format(algo, model_str), env=env)
    else:
        print("Algorythm not found")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=True)
print("\n\nModel: {}".format(algo + model_str))
print("Environment: {}".format(env_str))
print("Mean: {0:.3f}\n\n".format(mean_reward))

# Enjoy trained agent
test_env =  gym.make(env_str, max_steps=50, render_mode="human")
test_env = FlatObsWrapper(test_env)

observation = test_env.reset()[0]
for i in range(1000):
    action, _states = model.predict(observation, deterministic=False)
    observation, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        observation = test_env.reset()[0]