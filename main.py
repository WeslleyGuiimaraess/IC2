from Ambiente import Ambiente
from settings import *

def main():
    env = Ambiente() #carrega o ambiente
    env.inicia_ambiente_coop() #inicia o ambiente cooperativo
    for _ in range(EPISODIOS): #executa o número de espisódios
        env.executa_com_movimentos_aleatorios()
    env.encerra_ambiente()


if __name__ == "__main__":
        main()