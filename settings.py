import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

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

TEMPO_LIMITE    = 10000
PROGRESSO_FINAL = 640
RENDER          = True

EPISODIOS       = 5