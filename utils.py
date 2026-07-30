import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from random import sample

# Configuração centralizada (batch_size, resolution) vem de settings.py
from settings import batch_size, resolution

def preprocess(img):
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
   
    return tf.stack(img)


def split_tuple(samples):
    samples = np.array(samples, dtype=object)
    screen_buf = tf.stack(samples[:,0])
    actions = samples[:,1]
    rewards = tf.stack(samples[:,2])
    next_screen_buf = tf.stack(samples[:,3])
    dones = tf.stack(samples[:,4])  
    return screen_buf, actions, rewards, next_screen_buf, dones 


def extractDigits(*argv):
    if len(argv)==1:
        return list(map(lambda x: [x], argv[0]))

    return list(map(lambda x,y: [x,y], argv[0], argv[1]))


def get_samples(memory):
    if len(memory) < batch_size:
        sample_size = len(memory)
    else:
        sample_size = batch_size

    return sample(memory, sample_size)


# ação de exploração para o episódio de avaliação.
# vies_direita=True: prioriza andar para a direita (+ pular) para o agente atravessar
# a fase na demonstração; caso contrário, ação aleatória uniforme.
def acao_exploracao(num_actions, vies_direita=False):
    if vies_direita:
        r = np.random.uniform(0, 1)
        if r < 0.7:
            return 8   # andar para a direita (p1 RIGHT)
        if r < 0.85:
            return 9   # pular (p1 A) — ajuda a passar obstáculos
    return int(np.random.randint(num_actions))