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

OBJETOS = {
    'red': ['cacto', 'cachorro'],
    'blue': ['players'],
    'green': ['maca', 'caixa']
}


TEMPO_LIMITE    = 10000 #tempo de cada episodio
PROGRESSO_FINAL = 640 #valor que é registrado na memória quando o agente chega a um ponto x
RENDER          = True #ver a tela

EPISODIOS       = 5 #quantidade de episodios
