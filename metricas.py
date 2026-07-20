import os
import csv

import matplotlib
matplotlib.use('Agg')  # backend sem display (roda headless / Docker)
import matplotlib.pyplot as plt

from settings import RESULTADOS_DIR


# =========================================================================
# Coletor das métricas de desempenho do artigo (Seção 3.4.2):
#   - Tempo médio por frame (ms): pipeline de visão computacional + decisão
#   - Recompensa acumulada por episódio: R_E = Σ r_t
# A precisão na detecção de sprites (Eq. 3.1) é avaliada à parte, sobre um
# conjunto rotulado, via avaliar_precisao_deteccao (precisa de ground truth).
# =========================================================================
class ColetorMetricas:
    def __init__(self, abordagem, destino=None):
        self.abordagem = abordagem
        self.destino = destino or RESULTADOS_DIR
        self.episodios = []
        self._reset_frame()

    def _reset_frame(self):
        self._cv_ms = []
        self._dec_ms = []
        self._sprites = []

    def inicia_episodio(self):
        self._reset_frame()

    #registra o tempo (ms) de um frame: percepção (CV) + decisão do agente
    def registra_frame(self, cv_ms, dec_ms, sprites_detectados=None):
        self._cv_ms.append(cv_ms)
        self._dec_ms.append(dec_ms)
        if sprites_detectados is not None:
            self._sprites.append(sprites_detectados)

    #consolida as métricas do episódio
    def finaliza_episodio(self, episodio, recompensa):
        n = max(len(self._cv_ms), 1)
        soma_cv = sum(self._cv_ms)
        soma_dec = sum(self._dec_ms)
        registro = {
            'episodio': episodio,
            'recompensa': float(recompensa),
            'tempo_medio_frame_ms': (soma_cv + soma_dec) / n,
            'tempo_cv_ms': soma_cv / n,
            'tempo_decisao_ms': soma_dec / n,
            'sprites_detectados_medio': (sum(self._sprites) / len(self._sprites)) if self._sprites else 0.0,
        }
        self.episodios.append(registro)
        self._reset_frame()
        return registro

    #grava CSV + gráficos em resultados/<abordagem>_*
    def salvar(self):
        os.makedirs(self.destino, exist_ok=True)
        base = os.path.join(self.destino, self.abordagem)

        campos = ['episodio', 'recompensa', 'tempo_medio_frame_ms', 'tempo_cv_ms',
                  'tempo_decisao_ms', 'sprites_detectados_medio']
        caminho_csv = base + '_metricas.csv'
        with open(caminho_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            for r in self.episodios:
                w.writerow(r)

        eps = [r['episodio'] for r in self.episodios]
        rec = [r['recompensa'] for r in self.episodios]
        tmp = [r['tempo_medio_frame_ms'] for r in self.episodios]

        if eps:
            plt.figure()
            plt.plot(eps, rec, marker='o')
            plt.xlabel('Episódio'); plt.ylabel('Recompensa acumulada')
            plt.title(f'Recompensa por episódio - {self.abordagem}')
            plt.grid(True)
            plt.savefig(base + '_recompensa.png', bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.plot(eps, tmp, marker='o', color='tab:orange')
            plt.xlabel('Episódio'); plt.ylabel('Tempo médio por frame (ms)')
            plt.title(f'Tempo médio por frame - {self.abordagem}')
            plt.grid(True)
            plt.savefig(base + '_tempo_frame.png', bbox_inches='tight')
            plt.close()

        return caminho_csv


# =========================================================================
# Precisão na detecção de sprites (Eq. 3.1):
#   precisão = (detecções corretas / total de sprites) * 100
# Requer ground truth (contagem esperada por classe) rotulado à mão sobre um
# conjunto de frames de validação. Como só há contagens (sem bounding boxes),
# usa-se min(detectado, esperado) por classe como proxy de "detecções corretas".
# =========================================================================
def avaliar_precisao_deteccao(amostras):
    corretas = 0
    total = 0
    for detectado, esperado in amostras:
        for classe, n_esp in esperado.items():
            n_det = detectado.get(classe, 0)
            corretas += min(n_det, n_esp)
            total += n_esp
    if total == 0:
        return 0.0
    return 100.0 * corretas / total


# conta quantos sprites por classe o Transformador detectou num frame
def contar_deteccoes(transformador, frame):
    coords = transformador.extrair_coordenadas(frame)
    return {classe: len(pts) for classe, pts in coords.items()}
