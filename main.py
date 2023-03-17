from Ambiente import Ambiente
from settings import *

def main():
    env = Ambiente()
    env.inicia_ambiente_coop()
    for _ in range(EPISODIOS):
        env.executa_com_movimentos_aleatorios()
    env.encerra_ambiente()


if __name__ == "__main__":
        main()