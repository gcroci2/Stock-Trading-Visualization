from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('./data/MSFT.csv')
mode = 'live'

df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
# DummyVecEnv needs a function that returns an environment
# as an argument
# Stable Baselines algorithms expect the environment to be
# vectorized for consistency. Even if you are not using parallel
# environments (which would require a vectorized setup with
# multiple environments), using DummyVecEnv ensures that your
# single environment adheres to the expected interface.
# This is particularly useful for algorithms that are designed
# to take advantage of parallelism but need to work with a single
# environment during development or for simpler tasks.
env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode=mode)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50)
model.save("ppo_test")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_test")

obs = env.reset()
for i in range(len(df['Date'])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode=mode)
