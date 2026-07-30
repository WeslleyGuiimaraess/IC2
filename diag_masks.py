import cv2 as cv
from Transformador import Transformador

t = Transformador()
bgr = cv.imread('resultados/frames/aval_qlearning_ep00_f1584_bruto.png')
frame = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)  # como o render entrega

cv.imwrite('resultados/diag_cachorro.png', t.obter_cachorro(frame))
cv.imwrite('resultados/diag_segmentado.png', t.analisar(frame))
print('cachorro + segmentado regenerados')
