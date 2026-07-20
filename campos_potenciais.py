import numpy as np

from settings import *

# =========================================================================
# Controle reativo por instinto (Etapa 4 do artigo) — Campos Potenciais
# Artificiais. Quando todas as estimativas de Q ficam abaixo do limiar TAU,
# o agente ignora temporariamente a política aprendida e age por atração aos
# alvos (itens) e repulsão de ameaças (inimigos) e das bordas da tela.
# =========================================================================

# Encoding das ações discretas (compatível com _monta_action_list / train.py).
# Índice do botão pressionado = (action - 1) % 18. No NES:
#   6=DOWN, 7=LEFT, 8=RIGHT, 9=A(pulo). Ex.: RIGHT -> (8-1)=7.
ACAO_PEGAR    = 1   # botão B (pegar/arremessar)
ACAO_ABAIXAR  = 6   # DOWN
ACAO_ESQUERDA = 7   # LEFT
ACAO_DIREITA  = 8   # RIGHT
ACAO_PULAR    = 9   # A (pular / evadir)


def _mais_proximo(referencia, pontos):
    if not pontos:
        return None
    rx, ry = referencia
    return min(pontos, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2)


def _forca_atracao(player, alvo, k):
    if alvo is None:
        return np.array([0.0, 0.0])
    d = np.array([alvo[0] - player[0], alvo[1] - player[1]], dtype=float)
    dist = np.linalg.norm(d) + 1e-6
    return k * d / dist  # vetor unitário na direção do alvo


def _forca_repulsao(player, ameaca, k, alcance):
    if ameaca is None:
        return np.array([0.0, 0.0])
    d = np.array([player[0] - ameaca[0], player[1] - ameaca[1]], dtype=float)
    dist = np.linalg.norm(d) + 1e-6
    if dist > alcance:
        return np.array([0.0, 0.0])
    # repulsão cresce à medida que a ameaça se aproxima
    return k * (d / dist) * (alcance - dist) / alcance


def _forca_bordas(player, largura, altura, k, margem):
    x, y = player
    fx = fy = 0.0
    if x < margem:
        fx += (margem - x) / margem
    if x > largura - margem:
        fx -= (x - (largura - margem)) / margem
    if y < margem:
        fy += (margem - y) / margem
    if y > altura - margem:
        fy -= (y - (altura - margem)) / margem
    return np.array([fx, fy]) * k


def resultante(coords, largura, altura):
    # vetor de força resultante (fx, fy) no ponto do player
    players = coords.get('player') or []
    player = players[0] if players else (largura // 2, altura // 2)

    item = _mais_proximo(player, coords.get('item') or [])
    inimigo = _mais_proximo(player, coords.get('inimigo') or [])

    f = (_forca_atracao(player, item, CP_K_ATRACAO)
         + _forca_repulsao(player, inimigo, CP_K_REPULSAO, CP_ALCANCE_INIMIGO)
         + _forca_bordas(player, largura, altura, CP_K_BORDA, CP_MARGEM_BORDA))
    return player, item, inimigo, f


def acao_por_instinto(coords, largura, altura):
    # converte o vetor de força resultante em uma ação discreta do ambiente
    player, item, inimigo, f = resultante(coords, largura, altura)

    # inimigo muito próximo -> pula para evadir
    if inimigo is not None:
        dist = ((player[0] - inimigo[0]) ** 2 + (player[1] - inimigo[1]) ** 2) ** 0.5
        if dist < CP_DIST_EVASAO:
            return ACAO_PULAR

    fx, fy = float(f[0]), float(f[1])
    if abs(fx) < 1e-3 and abs(fy) < 1e-3:
        return ACAO_DIREITA  # sem forças relevantes -> progride para a direita

    # eixo dominante decide a ação (y cresce para baixo na imagem)
    if abs(fx) >= abs(fy):
        return ACAO_DIREITA if fx > 0 else ACAO_ESQUERDA
    return ACAO_PULAR if fy < 0 else ACAO_ABAIXAR
