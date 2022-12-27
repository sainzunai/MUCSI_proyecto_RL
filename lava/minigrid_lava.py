import sys
import gymnasium
sys.modules["gym"] = gymnasium

import gymnasium as gym

from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from minigrid.wrappers import FlatObsWrapper


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Create environment

env_str = "MiniGrid-MultiRoom-N6-v0"    # Prueba a meter 6 rooms sin entrenamiento previo

def create_env():
    env = gym.make(env_str, max_steps=100)
    env = FlatObsWrapper(env)
    return env

env = make_vec_env(create_env, n_envs=8)

continue_training = False

if continue_training:
    model = PPO.load("ppo", env=env)
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/{}".format(env_str))

# Train the agent and display a progress bar
model.learn(total_timesteps=int(3e7), reset_num_timesteps=(not continue_training)) #3e6
# Save the agent
model.save("ppo_{}".format(env_str))

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=True)
print(f"mean: {mean_reward}")

# Enjoy trained agent
test_env =  gym.make(env_str, max_steps=50, render_mode="human")
test_env = FlatObsWrapper(test_env)

observation = test_env.reset()[0]
for i in range(1000):
    action, _states = model.predict(observation, deterministic=False)
    observation, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        observation = test_env.reset()[0]

