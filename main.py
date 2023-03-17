import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

estado_anterior = {
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


def pega_recompensa_atual(estado_atual):
    return (
          (estado_atual['estrelas'] - estado_anterior['estrelas']) * 1000 \
        + (estado_atual['flores'] - estado_anterior['flores']) * 100 \
        + (estado_atual['progresso'] != 0) * 1 \
        + (estado_atual['tempo'] != 0) * -1 \
        + (estado_atual['1_coracao'] != 24) * -100 \
        + (estado_atual['2_coracao'] != 24) * -200 \
        + (estado_atual['3_coracao'] != 24) * -500 \
        + (estado_atual['pegar_jogar'] - estado_anterior['pegar_jogar']) * 1 \
        + (estado_atual['game_over'] != 0) * -1000 \
        + (estado_atual['mob'] - estado_anterior['mob']) * 1 \
    )

def main():
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        print("TicoTeco-Snes" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("TicoTeco-Snes", inttype=retro.data.Integrations.ALL, players=2)
        obs = env.reset()
        recompensa_total = 0
        progresso_atual = 0
        i = 0
        while True:

            #print(env.action_space.sample())

            # movimentos
            # [
            #   0:
            #   0:
            #   0:
            #   0:
            #   0: 
            #   0: baixo
            #   0: esquerda
            #   0: direita
            #   0: pula
            # ]
            p = 1 if i%5 != 0 else 0
            obs, rew, done, info = env.step(env.action_space.sample()+env.action_space.sample())
            i+=1
            recompensa_total += pega_recompensa_atual(info)
            estado_anterior = info

            progresso_atual += info['progresso']

            env.render()
            if progresso_atual == 640:
                obs = env.reset()
        env.close()


if __name__ == "__main__":
        main()