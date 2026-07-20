import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
# Estado e objetos do ambiente
# =========================================================================
ESTADO_INICIAL = {
    'estrelas'  : 128,
    'progresso' : 0,
    '2_coracao' : 24,
    '3_coracao' : 24,
    '1_coracao' : 24,
    'tempo'     : 0,
    'pegar_jogar': 0,
    'game_over' : 0,
    'flores'    : 128,
    'mob'       : 0
}

OBJETOS = {
    'red': ['cacto', 'cachorro'],
    'blue': ['players'],
    'green': ['maca', 'caixa']
}

# =========================================================================
# Controle de episódio
# =========================================================================
TEMPO_LIMITE    = 10000         # tempo (passos) máximo de cada episódio
PROGRESSO_FINAL = 160           # progresso que encerra o episódio (bate com scenario.json)
RENDER          = False         # exibir a tela do emulador

# =========================================================================
# Seleção da abordagem (Capítulo 3 do artigo)
#   'dqn_raw'      -> DQN sem filtro (baseline de pixels brutos)
#   'dqn_filtrado' -> DQN com filtro de cor (imagem segmentada)
#   'qlearning'    -> Q-Learning tabular (coordenadas -> grid -> Q-table)
# =========================================================================
ABORDAGEM = 'qlearning'

# Número de episódios de treino (fonte única; antes duplicado como
# num_train_epochs=50 em train.py e EPISODIOS=5 aqui)
EPISODIOS = 20  # config moderada p/ 1ª sessão ao vivo; artigo usa 50

# =========================================================================
# Pré-processamento / entrada da rede
# =========================================================================
resolution = (20, 30)           # (altura, largura) da imagem reduzida
model_savefolder = os.path.join(PROJECT_DIR, "model")
RESULTADOS_DIR   = os.path.join(PROJECT_DIR, "resultados")

# =========================================================================
# Captura de frames para as figuras da dissertação (PNG em resultados/frames/)
# =========================================================================
CAPTURAR_FRAMES   = False           # captura DURANTE o treino (cuidado: treino cheio gera muitos PNGs)
CAPTURA_INTERVALO = 30              # salva a cada N passos
FRAMES_DIR        = os.path.join(RESULTADOS_DIR, "frames")

# =========================================================================
# Episódio de avaliação após o treino ("assistir"): o agente joga a política
# aprendida de forma gulosa (sem exploração). Renderiza ao vivo se RENDER=True
# (precisa do VcXsrv) e salva os frames da avaliação em resultados/frames/.
# =========================================================================
ASSISTIR_APOS_TREINO = True
PASSOS_AVALIACAO     = 3000          # limite de passos do episódio de avaliação
RENDER_AVALIACAO     = True          # abre a janela ao vivo no episódio de avaliação (precisa do VcXsrv)
CAPTURAR_AVALIACAO   = True          # salva frame bruto + segmentado do episódio de avaliação (para as figuras)

# =========================================================================
# Hiperparâmetros do DQN
# =========================================================================
batch_size              = 64
learning_rate           = 0.00025
discount_factor         = 0.99
replay_memory_size      = 100000
learning_steps_per_epoch = 3000  # config moderada p/ 1ª sessão; artigo usa 10000
target_net_update_steps = 10

# Exploração (ε-greedy) do DQN
DQN_EPSILON       = 1.0
DQN_EPSILON_MIN   = 0.1
DQN_EPSILON_DECAY = 0.98

# =========================================================================
# Hiperparâmetros do Q-Learning tabular
# =========================================================================
QL_ALPHA         = 0.1          # taxa de aprendizado (α da Eq. 2.6)
QL_GAMMA         = 0.99         # fator de desconto (γ)
QL_EPSILON       = 1.0
QL_EPSILON_MIN   = 0.1
QL_EPSILON_DECAY = 0.9999

# Discretização das coordenadas em grade (Etapa 2 / Q-Learning)
GRID_W = 16
GRID_H = 16

# =========================================================================
# Chaveamento reativo por instinto (Etapa 4 - Campos Potenciais)
# =========================================================================
ATIVAR_INSTINTO = True          # habilita o override por campos potenciais
TAU             = 0.0           # limiar de utilidade: se max_a Q(s,a) < TAU -> instinto

# Ganhos e alcances dos campos potenciais (tunáveis)
CP_K_ATRACAO       = 1.0        # atração aos itens (maçãs/caixas)
CP_K_REPULSAO      = 2.0        # repulsão dos inimigos
CP_K_BORDA         = 1.0        # repulsão das bordas da tela
CP_ALCANCE_INIMIGO = 80.0       # raio (px) em que a repulsão do inimigo age
CP_MARGEM_BORDA    = 40.0       # margem (px) em que a borda começa a repelir
CP_DIST_EVASAO     = 40.0       # distância (px) do inimigo que força o pulo de evasão
