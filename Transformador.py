import cv2 as cv
import numpy as np

from Mascara import Mascara
from settings import *


class Transformador(object):

    def __init__(self):
        pass

    
    def analisar(self, input_image):
        _img = self.obter_cachorro(input_image)

        return _img

    def obter_maca(self, _img):

        lower_red = np.array([216, 0, 0])
        upper_red = np.array([216, 40, 0]) 

        kernel = np.ones((3,3), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        mask = cv.medianBlur(mask, 7)
        mask = cv.erode(mask, kernel)
        mask = np.stack((mask, mask, mask), axis=-1)

        _img = cv.bitwise_and(_img, mask)

        return _img


    def obter_caixa(self, _img):

        lower_red = np.array([200, 76, 8])
        upper_red = np.array([200, 76, 8]) 

        kernel = np.ones((5,5), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        mask = cv.medianBlur(mask, 3)
        mask = cv.erode(mask, kernel)
        mask = cv.dilate(mask, kernel, iterations=2)
        mask = np.stack((mask, mask, mask), axis=-1)

        _img = cv.bitwise_and(_img, mask)

        return _img


    def obter_player(self, _img):

        lower_red = np.array([248, 216, 168])
        upper_red = np.array([248, 216, 168]) 

        kernel = np.ones((2,2), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        mask = cv.medianBlur(mask, 5)
        mask = cv.erode(mask, kernel)
        mask = cv.dilate(mask, kernel, iterations=5)
        mask = np.stack((mask, mask, mask), axis=-1)

        _img = cv.bitwise_and(_img, mask)

        return _img


    def obter_cachorro(self, _img):

        lower_red = np.array([112, 120, 112])
        upper_red = np.array([112, 120, 112]) 

        kernel = np.ones((2,2), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        # mask = cv.medianBlur(mask, 3)
        # mask = cv.erode(mask, kernel)
        # mask = cv.dilate(mask, kernel, iterations=7)

        img_escala_cinza = cv.cvtColor(_img, cv.COLOR_BGR2GRAY)
        img_desfoque = cv.GaussianBlur(img_escala_cinza, (11, 11), 0)
        img_gradiente_contornos = cv.Canny(img_desfoque, 50, 100)
        sobel = cv.Sobel(img_escala_cinza, -1, 1, 1)

        _mask = cv.bitwise_and(sobel, mask, mask=None)
        combinando_tecnicas = np.stack((_mask, _mask, _mask), axis=-1)

        _img = cv.bitwise_and(_img, combinando_tecnicas)

        return _img