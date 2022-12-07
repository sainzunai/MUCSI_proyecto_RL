import gym
import os
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

ALGORTIHM="PPO_TEST"
models_dir = "Project_models/" + ALGORTIHM
log_dir = "Project_logs"



if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    

env = gym.make("ALE/MsPacman-v5", render_mode="human")  # render_mode="human" para ver el entrenamiento a velocidad "normal". Omitir este hiperparametro para que entrene mas rapido.

env.reset()



model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir) #TIMESTEPS = 200


TIMESTEPS = 1000 # Max. pasos sumatorio por cada .learn


for i in range(1, 1000000): # Episodios
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORTIHM)
    model.save(f"{models_dir}/{TIMESTEPS*i}.zip")

env.close()