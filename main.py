import gym
import tensorflow as tf
import itertools as it

from collections import deque
from time import sleep
from Ambiente import Ambiente
from settings import *

from utils import preprocess
from train import DQNAgent, run_dqn, assistir_dqn
from metricas import ColetorMetricas
import matplotlib.pyplot as plt

save_model = True
load = False
skip_learning = False  # True = modo "só avaliar" (carrega o modelo salvo e roda só a avaliação)
watch = False

# model_savefolder e replay_memory_size vêm de settings.py

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

def main():

    #inicia o ambiente do jogo
    env = Ambiente()
    env.inicia_ambiente_coop()

    print(f"Abordagem selecionada: {ABORDAGEM}")

    if skip_learning:
        #modo "só avaliar": carrega o modelo salvo e roda apenas o episódio de avaliação
        if ABORDAGEM in ('dqn_raw', 'dqn_filtrado'):
            n = env.env.action_space.n * 2
            agent = DQNAgent(num_actions=n, load=True)
            usar_filtro = (ABORDAGEM == 'dqn_filtrado')
            assistir_dqn(agent, env, usar_filtro=usar_filtro)
        return

    #coletor das métricas do artigo (tempo/frame, recompensa acumulada)
    coletor = ColetorMetricas(ABORDAGEM)

    if ABORDAGEM in ('dqn_raw', 'dqn_filtrado'):
        #pega a quantidade de ações que o agente pode executar no jogo
        n = env.env.action_space.n * 2

        #algoritmo que realiza aprendizado (agente)
        agent = DQNAgent(num_actions=n, load=load)
        replay_memory = deque(maxlen=replay_memory_size)

        usar_filtro = (ABORDAGEM == 'dqn_filtrado')
        x, y = run_dqn(agent, env, replay_memory, usar_filtro=usar_filtro, coletor=coletor)

        print(f'{x}\n{y}')

        if save_model:
            agent.dqn.save(model_savefolder)

        #após treinar: episódio de avaliação (agente joga guloso; você assiste)
        if ASSISTIR_APOS_TREINO:
            assistir_dqn(agent, env, usar_filtro=usar_filtro)

    elif ABORDAGEM == 'qlearning':
        # importado aqui para manter o DQN independente do Q-Learning tabular
        from qlearning import QLearningAgent, run_qlearning, assistir_qlearning

        n = env.env.action_space.n * 2
        agent = QLearningAgent(num_actions=n)
        x, y = run_qlearning(agent, env, coletor=coletor)

        print(f'{x}\n{y}')

        #após treinar: episódio de avaliação (agente joga guloso; você assiste)
        if ASSISTIR_APOS_TREINO:
            assistir_qlearning(agent, env)

    else:
        raise ValueError(f"ABORDAGEM desconhecida em settings.py: {ABORDAGEM!r}")

    caminho = coletor.salvar()
    print(f'Métricas salvas em: {caminho}')


if __name__ == '__main__':
    main()
