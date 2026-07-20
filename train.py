import numpy as np
import tensorflow as tf
from settings import *

from tqdm import trange
from time import time, sleep, perf_counter
from tensorflow.keras import Model, Sequential, Input, losses, metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, ReLU

from Transformador import Transformador
from campos_potenciais import acao_por_instinto
from captura import salvar_frames

import matplotlib.pyplot as plt

from utils import split_tuple, extractDigits, preprocess, get_samples

# Toda a configuração de aprendizado (batch_size, learning_rate, discount_factor,
# replay_memory_size, EPISODIOS, learning_steps_per_epoch, target_net_update_steps,
# model_savefolder, DQN_EPSILON*) vem de settings.py via 'from settings import *'.

#
class DQNAgent:
    def __init__(self, num_actions=9, epsilon=1, epsilon_min=0.1, epsilon_decay=0.98, load=False):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.optimizer = SGD(learning_rate)

        if load:
            print("Loading model from: ", model_savefolder) 
            self.dqn = tf.keras.models.load_model(model_savefolder)
        else:
            self.dqn = DQN(self.num_actions)
            self.target_net = DQN(self.num_actions)

    def update_target_net(self):
        self.target_net.set_weights(self.dqn.get_weights())
    
    #agente escolhe a ação dada a rede
    def choose_action(self, state):
        action = int(tf.argmax(self.dqn(tf.reshape(state, (1,20,30,1))), axis=1))

        return action

    #treina o agente
    def train_dqn(self, samples):
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

        #aplica o algoritmo de Q learning
        with tf.GradientTape() as tape:
            tape.watch(self.dqn.trainable_variables)

            Q_prev = tf.gather_nd(self.dqn(screen_buf), ids)
            
            Q_next = self.target_net(next_screen_buf)
            Q_next = tf.gather_nd(Q_next, extractDigits(row_ids, tf.argmax(self.dqn(next_screen_buf), axis=1)))
            
            q_target = rewards + self.discount_factor * Q_next

            if len(done_ids)>0:
                done_rewards = tf.gather_nd(rewards, done_ids)
                q_target = tf.tensor_scatter_nd_update(tensor=q_target, indices=done_ids, updates=done_rewards)

            td_error = tf.keras.losses.MSE(q_target, Q_prev)

        gradients = tape.gradient(td_error, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


#rede neural
class DQN(Model):
    #define a estrutura da rede
    def __init__(self, num_actions):
        super(DQN,self).__init__()
        self.conv1 = Sequential([
                                Conv2D(8, kernel_size=6, strides=3, input_shape=(20,30,1)),
                                BatchNormalization(),
                                ReLU()
                                ])

        self.conv2 = Sequential([
                                Conv2D(8, kernel_size=3, strides=2, input_shape=(5, 9, 8)),
                                BatchNormalization(),
                                ReLU()
                                ])
        
        self.flatten = Flatten()
       
        self.state_value = Dense(1) 
        self.advantage = Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        # dueling: metade das features vai para o fluxo de valor e metade para o de
        # vantagem. O flatten produz 64 valores; o split anterior em 96 deixava o
        # fluxo de vantagem vazio (bug), tornando a ação escolhida independente do estado.
        x1 = x[:, :32]
        x2 = x[:, 32:]
        x1 = self.state_value(x1)
        x2 = self.advantage(x2)
        
        x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1,1)))
        return x


#processa o frame de acordo com a abordagem escolhida
def processa_frame(t, frame, usar_filtro):
    # usar_filtro=True  -> imagem segmentada por cor (DQN com filtro)
    # usar_filtro=False -> pixels brutos, sem filtro (baseline DQN sem filtro)
    if usar_filtro:
        frame = t.analisar(frame)
    return preprocess(frame)


#teina o agente no ambiente(jogo)
def run_dqn(agent, env, replay_memory, usar_filtro=True, coletor=None):

    t = Transformador()

    time_start = time()

    x = []
    y = []
    #contabiliza a quantidade de iterações do agente dentro ambiente
    for episode in range(EPISODIOS):
        train_scores = []
        print("\nEpoch %d\n-------" % (episode + 1))

        total_reward = 0

        env.reinicia_ambiente()
        if coletor: coletor.inicia_episodio()

        # frame_atual: frame bruto correspondente ao screen_buf (usado pelo instinto)
        cv_ini = perf_counter()
        frame_atual = env.env.render(mode='rgb_array')
        altura, largura = frame_atual.shape[0], frame_atual.shape[1]
        next_screen_buf = processa_frame(t, frame_atual, usar_filtro)
        cv_ms = (perf_counter() - cv_ini) * 1000.0
        action = env.env.action_space.sample() + env.env.action_space.sample()

        env.env.step(action)
        #skipa os frames para tomar a decisão
        for i in trange(learning_steps_per_epoch, leave=False):
            #trata a imagem cor/tamanho da iamgem para que possa ser utilizada
            screen_buf = next_screen_buf

            #salva frames (bruto + segmentado) para as figuras da dissertação
            if CAPTURAR_FRAMES and (i % CAPTURA_INTERVALO == 0):
                salvar_frames(ABORDAGEM, episode, i, frame_atual, t)

            dec_ini = perf_counter()
            if agent.epsilon < np.random.uniform(0,1):
                q_values = agent.dqn(tf.reshape(screen_buf, (1,) + resolution + (1,)))
                action = int(tf.argmax(q_values, axis=1))
                # Etapa 4: chaveamento reativo por instinto quando a utilidade é baixa
                if ATIVAR_INSTINTO and float(tf.reduce_max(q_values)) < TAU:
                    coords = t.extrair_coordenadas(frame_atual)
                    action = acao_por_instinto(coords, largura, altura)
            else:
                if 0.8 > np.random.uniform(0,1):
                    if 0.5 < np.random.uniform(0,1):
                        action = np.random.choice(range(17, 19), 1)[0]
                        #action = 17
                    else:
                        action = np.random.choice(range(8, 10), 1)[0]
                        #action = 8
                else:
                    action = np.random.choice(range(env.env.action_space.n * 2), 1)[0]
            dec_ms = (perf_counter() - dec_ini) * 1000.0
            #registra o tempo do frame (percepção + decisão)
            if coletor: coletor.registra_frame(cv_ms, dec_ms)

            #observa a ação tomado pelo agente para poder dar a recompensa
            action_list = [1 if i==((action-1)%18) else 0 for i in range(env.env.action_space.n)]
            action_list += [1 if i==((action-1)%18) else 0 for i in range(env.env.action_space.n)]

            observation, reward, done, info = env.env.step(action_list)
            env.estado_atual = info #pega informação atual do ambiente
            env.progresso_atual    += info['progresso'] #atualiza o progresso

            #soma o valor da recompensa acumulando o total
            reward = float(env.pega_recompensa_atual())

            total_reward += reward

            env.tempo_atual += 1

            env.estado_anterior = env.estado_atual #atualiza o estadual anterior
            if RENDER:  env.env.render() #permite a exibição da cena em modo gráfico

            if (env.progresso_atual > PROGRESSO_FINAL) or (env.tempo_atual > TEMPO_LIMITE):
                done = True

            #pega o proximo frame para o agente tomar a decisão (cronometra o pipeline de CV)
            if not done:
                cv_ini = perf_counter()
                frame_atual = env.env.render(mode='rgb_array')
                next_screen_buf = processa_frame(t, frame_atual, usar_filtro)
                cv_ms = (perf_counter() - cv_ini) * 1000.0
            else:
                next_screen_buf = tf.zeros(shape=screen_buf.shape)
                cv_ms = 0.0

            #caso o agente atinja o objetivo o agente reseta tudo e adiciona a recompensa 
            if done:
                train_scores.append(total_reward)
                env.reinicia_ambiente()

            #armazena todos os dados da tela, ação, recompensa, proxima tela
            replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

            #se o frame atual for maior que batch, o algoritmo começa a treinar o agente
            if i % batch_size == 0:
                agent.train_dqn(get_samples(replay_memory))
       
            #
            if ((i % target_net_update_steps) == 0):
                agent.update_target_net()
            
        print(f'Total score episode {episode}: {total_reward}')
        x.append(episode)
        y.append(total_reward)
        if coletor: coletor.finaliza_episodio(episode, total_reward)
        agent.dqn.save_weights(f'./model_{episode}')

        train_scores = np.array(train_scores)

    return x, y


#após o treino: episódio de avaliação com o agente jogando guloso (política aprendida)
def assistir_dqn(agent, env, usar_filtro=True):
    t = Transformador()
    env.reinicia_ambiente()

    frame = env.env.render(mode='rgb_array')
    screen_buf = processa_frame(t, frame, usar_filtro)

    total_reward = 0
    done = False
    passo = 0
    ao_vivo = RENDER_AVALIACAO
    print("\n=== Episódio de avaliação (assistir) ===")
    while not done and passo < PASSOS_AVALIACAO:
        #ação gulosa: sempre o maior Q (sem exploração)
        action = int(tf.argmax(agent.dqn(tf.reshape(screen_buf, (1,) + resolution + (1,))), axis=1))

        action_list = [1 if k == ((action-1)%18) else 0 for k in range(env.env.action_space.n)]
        action_list += [1 if k == ((action-1)%18) else 0 for k in range(env.env.action_space.n)]

        observation, reward, done, info = env.env.step(action_list)
        env.estado_atual = info
        env.progresso_atual += info['progresso']
        total_reward += float(env.pega_recompensa_atual())
        env.tempo_atual += 1
        env.estado_anterior = env.estado_atual

        if ao_vivo:  #janela ao vivo (precisa do VcXsrv); best-effort
            try:
                env.env.render()
            except Exception as e:
                print(f"[aviso] janela ao vivo indisponível ({e}); seguindo com os frames salvos.")
                ao_vivo = False
        if CAPTURAR_AVALIACAO and (passo % CAPTURA_INTERVALO == 0):
            salvar_frames(f'aval_{ABORDAGEM}', 0, passo, frame, t)

        if (env.progresso_atual > PROGRESSO_FINAL) or (env.tempo_atual > TEMPO_LIMITE):
            done = True

        if not done:
            frame = env.env.render(mode='rgb_array')
            screen_buf = processa_frame(t, frame, usar_filtro)
        passo += 1

    print(f'Recompensa do episódio de avaliação: {total_reward}')
    return total_reward