class Mascara(object):

    def __init__(self):
        self._identificador = None
        self._img_filtrada  = None
        self._tipo          = None

    
    def criar(self, _id, _img, _tipo):
        self._identificador = _id
        self._img_filtrada  = _img
        self._tipo          = _tipo