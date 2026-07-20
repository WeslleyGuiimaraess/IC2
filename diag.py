import numpy as np
from Ambiente import Ambiente

env = Ambiente()
env.inicia_ambiente_coop()
env.reinicia_ambiente()

n = env.env.action_space.n

def monta(action, nb):
    lst = [1 if k == ((action - 1) % 18) else 0 for k in range(nb)]
    lst += [1 if k == ((action - 1) % 18) else 0 for k in range(nb)]
    return lst

ram = env.env.get_ram()
print("tamanho RAM:", ram.shape)

# compara o endereço atual de 'tempo' (0) com o do artigo (0x0411=1041) ao longo dos passos
print("\npasso | ram[0] | ram[1041] | ram[25](progr) | ram[1458](estr) | ram[528](cor1)")
for i in range(60):
    env.env.step(monta(8, n))  # andar para a direita
    ram = env.env.get_ram()
    if i % 3 == 0:
        print(f"{i:5d} | {ram[0]:5d} | {ram[1041]:8d} | {ram[25]:6d} | {ram[1458]:6d} | {ram[528]:6d}")

env.encerra_ambiente()
