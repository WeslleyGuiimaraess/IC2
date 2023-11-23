import gym
import tensorflow as tf
import itertools as it

from collections import deque
from time import sleep
from Ambiente import Ambiente
from settings import *

from utils import preprocess
from train import DQNAgent, run
import matplotlib.pyplot as plt

save_model = True
load = False
skip_learning = False
watch = False

model_savefolder = "./model"

replay_memory_size = 1000000


tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# def main():
    
#     #inicia o ambiente do jogo
#     env = Ambiente()
#     env.inicia_ambiente_coop()
#     #pega a quantidade de ações que o agente pode executar no jogo
#     n = env.env.action_space.n * 2
    
#     #algoritmo que realiza aprendizado (agente)
#     agent = DQNAgent(num_actions=n, load=load)

#     replay_memory = deque(maxlen=replay_memory_size)
    
#     #print(env.render(mode='rgb_array').shape)
#     if not skip_learning:
#         x, y = run(agent, env, replay_memory)

#         print(f'{x}\n{y}')

#         if save_model:
#             agent.dqn.save(model_savefolder)
    

if __name__ == '__main__':
#    main()
    env = Ambiente()
    env.inicia_ambiente_coop()
    n = env.env.action_space.n
    env.executa_com_movimentos_aleatorios()