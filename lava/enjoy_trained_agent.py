import sys
import gymnasium
sys.modules["gym"] = gymnasium

import gymnasium as gym

from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from minigrid.wrappers import FlatObsWrapper


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env_str = 'MiniGrid-Empty-5x5-v0'
model_str = env_str # Dejar asi para evaluar al agente en el mismo mapa que se ha entrenado



def create_env():
    env = gym.make(env_str, max_steps=100)
    env = FlatObsWrapper(env)
    return env

env = make_vec_env(create_env, n_envs=8)

model = PPO.load("models/ppo_{}".format(model_str), env=env)

# Enjoy trained agent
test_env =  gym.make(env_str, max_steps=50, render_mode="human")
test_env = FlatObsWrapper(test_env)

observation = test_env.reset()[0]
for i in range(1000):
    action, _states = model.predict(observation, deterministic=False)
    observation, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        observation = test_env.reset()[0]