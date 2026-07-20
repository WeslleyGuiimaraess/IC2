import os
import cv2 as cv

from settings import FRAMES_DIR


# =========================================================================
# Captura de frames para as figuras da dissertação.
# Salva, a cada N passos (ver CAPTURA_INTERVALO em settings.py):
#   - o frame bruto do emulador;
#   - a imagem segmentada por cor (players em azul, itens em verde,
#     inimigos em vermelho) — mesmo padrão das Figuras 13/14 do artigo.
# Funciona headless (não precisa de display); grava PNG em resultados/frames/.
# =========================================================================
def salvar_frames(abordagem, episodio, passo, frame_rgb, transformador):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    prefixo = os.path.join(FRAMES_DIR, f"{abordagem}_ep{episodio:02d}_f{passo:04d}")

    # frame bruto: render(mode='rgb_array') devolve RGB; o OpenCV grava em BGR
    cv.imwrite(prefixo + "_bruto.png", cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))

    # imagem segmentada: analisar() já devolve no formato (B, G, R) esperado pelo imwrite
    segmentado = transformador.analisar(frame_rgb)
    cv.imwrite(prefixo + "_segmentado.png", segmentado)
