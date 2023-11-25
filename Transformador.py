import cv2 as cv
import numpy as np

from Mascara import Mascara
from settings import *


class Transformador(object):

    def __init__(self):
        pass

    
    def analisar(self, input_image):

        _img_maca = self.obter_maca(input_image)
        _img_caixa = self.obter_caixa(input_image)
        _img_player = self.obter_player(input_image)
        _img_cachoro = self.obter_cachorro(input_image)

        r, g, b = input_image[:, :, 2], input_image[:, :, 1], input_image[:, :, 0]

        _mask_green = cv.bitwise_or(_img_maca, _img_caixa)
        _mask_blue  = _img_player
        _mask_red   = _img_cachoro

        _r = cv.bitwise_and(r, _mask_red)
        _g = cv.bitwise_and(g, _mask_green)
        _b = cv.bitwise_and(b, _mask_blue)

        _img = np.stack((_b, _g, _r), axis=-1)

        return _img


    def obter_maca(self, _img):

        lower_red = np.array([216, 0, 0])
        upper_red = np.array([216, 40, 0]) 

        kernel = np.ones((3,3), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        mask = cv.medianBlur(mask, 7)
        mask = cv.erode(mask, kernel)

        return mask


    def obter_caixa(self, _img):

        lower_red = np.array([200, 76, 8])
        upper_red = np.array([200, 76, 8]) 

        kernel = np.ones((5,5), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        mask = cv.medianBlur(mask, 3)
        mask = cv.erode(mask, kernel)
        mask = cv.dilate(mask, kernel, iterations=2)

        return mask


    def obter_player(self, _img):

        lower_red = np.array([248, 216, 168])
        upper_red = np.array([248, 216, 168]) 

        kernel = np.ones((2,2), np.uint8)
        mask = cv.inRange(_img, lower_red, upper_red)

        mask = cv.medianBlur(mask, 5)
        mask = cv.erode(mask, kernel)
        mask = cv.dilate(mask, kernel, iterations=13)

        return mask


    def obter_cachorro(self, _img):

        lower_gray = np.array([184, 188, 184])
        upper_gray = np.array([184, 188, 184]) 

        kernel = np.ones((9,9), np.uint8)
        mask = cv.inRange(_img, lower_gray, upper_gray)

        mask = cv.medianBlur(mask, 7)
        mask = cv.erode(mask, kernel)
        mask = cv.dilate(mask, kernel, iterations=5)

        _img_gray = cv.cvtColor(_img, cv.COLOR_BGR2GRAY)
        sobel = cv.Sobel(_img_gray, -1, 0, 1)

        kernel = np.ones((4, 4), np.uint8)
        _mask = cv.bitwise_and(sobel, mask, mask=None)
        _mask = cv.erode(_mask, kernel)
        _mask = cv.dilate(_mask, kernel, iterations=5)
        ret, _mask = cv.threshold(_mask, 1, 255, cv.THRESH_BINARY)

        return _mask 