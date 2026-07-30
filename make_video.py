import re
import glob
from PIL import Image

from settings import ABORDAGEM

# monta um GIF lado a lado (frame bruto | imagem segmentada) do episódio de avaliação
# da abordagem selecionada em settings.py (ABORDAGEM).
def num(f):
    m = re.search(r'_f(\d+)_', f)
    return int(m.group(1)) if m else 0

brutos = sorted(glob.glob(f'resultados/frames/aval_{ABORDAGEM}_*_bruto.png'), key=num)
segs = sorted(glob.glob(f'resultados/frames/aval_{ABORDAGEM}_*_segmentado.png'), key=num)
print(f'[{ABORDAGEM}] frames:', len(brutos), 'bruto /', len(segs), 'segmentado')

quadros = []
for b, s in zip(brutos, segs):
    ib = Image.open(b).convert('RGB')
    isg = Image.open(s).convert('RGB')
    canvas = Image.new('RGB', (ib.width + isg.width, max(ib.height, isg.height)))
    canvas.paste(ib, (0, 0))
    canvas.paste(isg, (ib.width, 0))
    quadros.append(canvas)

if quadros:
    saida = f'resultados/{ABORDAGEM}_avaliacao.gif'
    quadros[0].save(saida, save_all=True, append_images=quadros[1:], duration=100, loop=0)  # 10 fps
    print('GIF salvo em', saida)
else:
    print('nenhum frame encontrado para', ABORDAGEM)
