import numpy as np

from collections import defaultdict
from time import perf_counter
from tqdm import trange

from settings import *
from Transformador import Transformador
from campos_potenciais import acao_por_instinto
from captura import salvar_frames


# =========================================================================
# Discretização do espaço de estados (Etapa 2 do artigo)
#   As coordenadas contínuas (x, y) dos objetos são mapeadas em uma grade
#   GRID_W x GRID_H, reduzindo a dimensionalidade para uma Q-table tratável.
# =========================================================================
def discretiza(ponto, largura, altura):
    x, y = ponto
    cx = min(int(x / max(largura, 1) * GRID_W), GRID_W - 1)
    cy = min(int(y / max(altura, 1) * GRID_H), GRID_H - 1)
    return (cx, cy)


def _mais_proximo(referencia, pontos):
    if not pontos:
        return None
    rx, ry = referencia
    return min(pontos, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2)


def monta_estado(coords, largura, altura):
    # estado = (célula do player, célula do item mais próximo, célula do inimigo mais próximo)
    players = coords.get('player') or []
    player = players[0] if players else (largura // 2, altura // 2)

    item = _mais_proximo(player, coords.get('item') or [])
    inimigo = _mais_proximo(player, coords.get('inimigo') or [])

    cel_player = discretiza(player, largura, altura)
    cel_item = discretiza(item, largura, altura) if item else (-1, -1)
    cel_inimigo = discretiza(inimigo, largura, altura) if inimigo else (-1, -1)

    return (cel_player, cel_item, cel_inimigo)


# =========================================================================
# Agente Q-Learning tabular
# =========================================================================
class QLearningAgent:
    def __init__(self, num_actions, alpha=QL_ALPHA, gamma=QL_GAMMA,
                 epsilon=QL_EPSILON, epsilon_min=QL_EPSILON_MIN,
                 epsilon_decay=QL_EPSILON_DECAY):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table esparsa: estado -> vetor de valores Q por ação
        self.q_table = defaultdict(lambda: np.zeros(num_actions, dtype=np.float32))

    def valores_q(self, estado):
        return self.q_table[estado]

    #agente escolhe a ação (ε-greedy)
    def choose_action(self, estado):
        if np.random.uniform(0, 1) < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[estado]))

    #atualiza a Q-table pela equação de Bellman (Eq. 2.6)
    def update(self, estado, acao, recompensa, prox_estado, done):
        q_atual = self.q_table[estado][acao]
        if done:
            alvo = recompensa
        else:
            alvo = recompensa + self.gamma * np.max(self.q_table[prox_estado])
        self.q_table[estado][acao] = q_atual + self.alpha * (alvo - q_atual)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


# constrói o vetor de ação (coop, 2 jogadores) — mesmo encoding usado em train.py
def _monta_action_list(action, n_botoes):
    lst = [1 if k == ((action - 1) % 18) else 0 for k in range(n_botoes)]
    lst += [1 if k == ((action - 1) % 18) else 0 for k in range(n_botoes)]
    return lst


#treina o agente Q-Learning tabular no ambiente
def run_qlearning(agent, env, coletor=None):

    t = Transformador()

    x = []
    y = []

    # dimensões do frame para a discretização (altura, largura)
    frame = env.env.render(mode='rgb_array')
    altura, largura = frame.shape[0], frame.shape[1]

    for episode in range(EPISODIOS):
        print("\nEpoch %d\n-------" % (episode + 1))

        total_reward = 0
        env.reinicia_ambiente()
        if coletor: coletor.inicia_episodio()

        cv_ini = perf_counter()
        frame = env.env.render(mode='rgb_array')
        coords_atual = t.extrair_coordenadas(frame)
        estado = monta_estado(coords_atual, largura, altura)
        cv_ms = (perf_counter() - cv_ini) * 1000.0
        frame_atual = frame  # frame bruto correspondente ao estado atual (para captura)

        for i in trange(learning_steps_per_epoch, leave=False):
            #salva frames (bruto + segmentado) para as figuras da dissertação
            if CAPTURAR_FRAMES and (i % CAPTURA_INTERVALO == 0):
                salvar_frames(ABORDAGEM, episode, i, frame_atual, t)

            # seleção de ação (ε-greedy) com chaveamento reativo por instinto (Etapa 4)
            dec_ini = perf_counter()
            if np.random.uniform(0, 1) < agent.epsilon:
                action = int(np.random.randint(agent.num_actions))
            elif ATIVAR_INSTINTO and float(np.max(agent.valores_q(estado))) < TAU:
                action = acao_por_instinto(coords_atual, largura, altura)
            else:
                action = int(np.argmax(agent.valores_q(estado)))
            dec_ms = (perf_counter() - dec_ini) * 1000.0
            #registra tempo do frame + nº de sprites detectados (percepção)
            if coletor:
                n_sprites = sum(len(v) for v in coords_atual.values())
                coletor.registra_frame(cv_ms, dec_ms, sprites_detectados=n_sprites)

            action_list = _monta_action_list(action, env.env.action_space.n)

            observation, reward, done, info = env.env.step(action_list)
            env.estado_atual = info
            env.progresso_atual += info['progresso']

            reward = float(env.pega_recompensa_atual())
            total_reward += reward

            env.tempo_atual += 1
            env.estado_anterior = env.estado_atual
            if RENDER:  env.env.render()

            if (env.progresso_atual > PROGRESSO_FINAL) or (env.tempo_atual > TEMPO_LIMITE):
                done = True

            if not done:
                cv_ini = perf_counter()
                prox_frame = env.env.render(mode='rgb_array')
                prox_coords = t.extrair_coordenadas(prox_frame)
                prox_estado = monta_estado(prox_coords, largura, altura)
                cv_ms = (perf_counter() - cv_ini) * 1000.0
            else:
                prox_coords = coords_atual
                prox_estado = estado
                cv_ms = 0.0

            agent.update(estado, action, reward, prox_estado, done)
            estado = prox_estado
            coords_atual = prox_coords
            if not done:
                frame_atual = prox_frame

            if done:
                env.reinicia_ambiente()
                frame = env.env.render(mode='rgb_array')
                coords_atual = t.extrair_coordenadas(frame)
                estado = monta_estado(coords_atual, largura, altura)
                frame_atual = frame

        print(f'Total score episode {episode}: {total_reward}')
        x.append(episode)
        y.append(total_reward)
        if coletor: coletor.finaliza_episodio(episode, total_reward)

    return x, y


#após o treino: episódio de avaliação com o agente jogando guloso (política aprendida)
def assistir_qlearning(agent, env):
    t = Transformador()
    env.reinicia_ambiente()

    frame = env.env.render(mode='rgb_array')
    altura, largura = frame.shape[0], frame.shape[1]
    coords = t.extrair_coordenadas(frame)
    estado = monta_estado(coords, largura, altura)

    total_reward = 0
    done = False
    passo = 0
    ao_vivo = RENDER_AVALIACAO
    print("\n=== Episódio de avaliação (assistir) ===")
    while not done and passo < PASSOS_AVALIACAO:
        action = int(np.argmax(agent.valores_q(estado)))  #ação gulosa (sem exploração)

        action_list = _monta_action_list(action, env.env.action_space.n)

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
            coords = t.extrair_coordenadas(frame)
            estado = monta_estado(coords, largura, altura)
        passo += 1

    print(f'Recompensa do episódio de avaliação: {total_reward}')
    return total_reward
