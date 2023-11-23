import retro
import cv2 as cv
import numpy as np
from time import sleep

from settings import *
from Transformador import Transformador

class Ambiente(object):

    def __init__(self):
        self.estado_anterior    = ESTADO_INICIAL
        self.estado_atual       = ESTADO_INICIAL
        self.env                = None
        self.observation        = None

        self.recompensa_total   = None
        self.tempo_atual        = None
        self.progresso_atual    = None

        self.tipo_ambiente      = None # 'COOP' ou 'SING'

        retro.data.Integrations.add_custom_path(
            os.path.join(PROJECT_DIR, "custom_integrations")
        ) #adicionando o diretorio do jogo

    def inicia_ambiente_singleplayer(self):
        self.env = retro.make("TicoTeco-Snes", inttype=retro.data.Integrations.ALL, players=1)
        self.obs = self.env.reset()
        self.tipo_ambiente = 'SING'


    def inicia_ambiente_coop(self):
        self.env = retro.make("TicoTeco-Snes", inttype=retro.data.Integrations.ALL, players=2)
        self.obs = self.env.reset()
        self.tipo_ambiente = 'COOP'


    def reinicia_ambiente(self):
        self.obs                = self.env.reset()
        self.estado_anterior    = ESTADO_INICIAL
        self.estado_atual       = ESTADO_INICIAL
        self.recompensa_total   = 0
        self.tempo_atual        = 0
        self.progresso_atual    = 0


    def pega_recompensa_atual(self):
        return (
            (self.estado_atual['estrelas'] - self.estado_anterior['estrelas']) * 1000 \
            + (self.estado_atual['flores'] - self.estado_anterior['flores']) * 200 \
            + (self.estado_atual['progresso'] != 0) * 1 \
            + (self.estado_atual['tempo'] != 0) * -1 \
            + (self.estado_atual['1_coracao'] != 24) * -100 \
            + (self.estado_atual['2_coracao'] != 24) * -200 \
            + (self.estado_atual['3_coracao'] != 24) * -500 \
            + (self.estado_atual['pegar_jogar'] - self.estado_anterior['pegar_jogar']) * 1 \
            + (self.estado_atual['game_over'] != 0) * -1000 \
            + (self.estado_atual['mob'] - self.estado_anterior['mob']) * 1 \
        )


    def executa_com_movimentos_aleatorios(self):

        t = Transformador()

        if self.env is None: return #verifica se há modo selecionado

        self.reinicia_ambiente()

        done = False 
        while(not done):
            
            action = [0, 0, 0, 0, 0, 0, 0, 1, 0] #escolhe ação aleátoria
            if self.tipo_ambiente == 'COOP':
                action += [0, 0, 0, 0, 0, 0, 0, 1, 0] # Permite movimentação do 2° jogador

            obs, rew, done, info = self.env.step(action) #executa a ação e retorna o resultado da mesma
            self.estado_atual = info #pega informação atual do ambiente
            self.recompensa_total   += self.pega_recompensa_atual() #calcula a recompensa da ação gerada
            self.tempo_atual        += 1 #atualiza o tempo
            self.progresso_atual    += info['progresso'] #atualiza o progresso
            
            self.estado_anterior = self.estado_atual #atualiza o estadual anterior

            if RENDER:
                _img = self.env.render(mode='rgb_array') #permite a exibição da cena em modo gráfico
                _img2 = t.analisar(_img)

                cv.imshow('Video Player', _img2)
                cv.imshow('original', cv.cvtColor(_img * 2, cv.COLOR_BGR2RGB))
                if cv.waitKey(25) & 0xFF == ord('q'):
                    exit(0)

            if (self.progresso_atual > PROGRESSO_FINAL) or (self.tempo_atual > TEMPO_LIMITE):
                done = True
            
            print(self.recompensa_total)


    def encerra_ambiente(self):
        self.env.close()