# Correções de texto do artigo (alinhar descrição ao código)

Decisão: **manter o pipeline de visão computacional do código** (segmentação por
`cv.inRange` sobre os canais RGB + morfologia; inimigo por **Sobel**) e **corrigir o texto**
do artigo, que hoje descreve HSV e Canny — técnicas que o código **não** usa.

O que o código realmente faz (`Transformador.py`):
- Segmentação de cor (`obter_maca`, `obter_caixa`, `obter_player`, `obter_cachorro`):
  `cv.inRange` **direto sobre os canais RGB brutos** (sem conversão para HSV) + `medianBlur`,
  `erode`, `dilate`.
- Inimigo: máscara de cor combinada com **`cv.Sobel`** (derivada vertical, `cv.Sobel(gray,-1,0,1)`)
  e limiarização (`cv.threshold`). **Não** há `Canny` nem `GaussianBlur`.
- Extração de coordenadas (`extrair_coordenadas`): `cv.findContours` + `cv.moments` para obter
  os centróides de cada objeto.

> Preciso do **fonte LaTeX (`.tex`)** para aplicar as edições abaixo diretamente. Com o PDF só
> consigo indicar os trechos. Enquanto isso, seguem os textos corrigidos para colar.

---

## 1) Seção 3.3.2 "Utilizando Visão Computacional" — espaço de cor

**Onde:** "...uma técnica de detecção por cor (segmentação cromática) para mapear os objetos de
interesse..." e a frase que cita o **espaço HSV**.

**Antes (trecho):**
> ...utiliza-se a técnica de limiarização de cores, que segmenta os objetos de interesse com
> base em suas características de cor **no espaço HSV**.

**Depois:**
> ...utiliza-se a técnica de limiarização de cores (`cv.inRange`), que segmenta os objetos de
> interesse aplicando limiares **diretamente sobre os canais de cor RGB** do quadro, sem
> conversão para outro espaço de cor.

## 2) Seção 3.3.2 — funções de "detecção por borda"

**Antes:** a lista de funções de detecção por borda cita **GaussianBlur**, **Canny** e
**findContours**.

**Depois (substituir a lista):**
> Na técnica de detecção por borda utiliza-se o operador de **Sobel** (`cv.Sobel`, derivada
> na direção vertical) aplicado sobre a imagem em tons de cinza e combinado à máscara de cor,
> seguido de limiarização (`cv.threshold`) para isolar o inimigo. A partir das máscaras
> resultantes, a função **`cv.findContours`** e os **momentos** (`cv.moments`) são usados para
> obter os centróides (coordenadas) de cada objeto.

(Ou seja: remover **GaussianBlur** e **Canny**; manter **findContours**, agora no papel de
extração de centróides; acrescentar **Sobel** e **threshold**.)

## 3) Seção 3.4.1, Etapa 1 "Detecção e Segmentação de Objetos via OpenCV"

**Antes:**
> - Maçãs e Caixotes: ...limiarização de cores, que segmenta os objetos de interesse com base
>   em suas características de cor **no espaço HSV**.
> - Inimigos: Para a detecção dos inimigos será utilizado **filtro de borda** para identificar
>   contornos e formas relevantes...

**Depois:**
> - Maçãs e Caixotes: limiarização de cores (`cv.inRange`) aplicada **sobre os canais RGB**,
>   seguida de operações morfológicas (`medianBlur`, `erode`, `dilate`).
> - Inimigos: detecção por **operador de Sobel** (derivada vertical) combinada à máscara de
>   cor e limiarização, delimitando a região do inimigo.
> - Itens Coletáveis: combinação de segmentação por cor (RGB) e análise de contornos
>   (`findContours` + `moments`) para localizar os centróides.

## 4) Consistência do Resumo/Abstract (opcional)

O Resumo cita **YOLO** como algoritmo de detecção usado com as duas abordagens. O código **não**
usa YOLO — a detecção é feita por segmentação de cor + Sobel (visão computacional clássica).
Recomenda-se remover a menção a YOLO do Resumo/Abstract ou reformular para "técnicas clássicas
de visão computacional (segmentação por cor e detecção de bordas)", alinhando à Seção 3.

---

## Observações sobre outros pontos já alinhados no código (não precisam de correção)

- **Tabela 1 (pontuação por ação)**: o código foi ajustado para bater exatamente com a tabela
  (flor +100, 2º coração −500, 3º coração −1000, derrotar inimigo +500, pegar/arremessar +10).
- **Três abordagens** (DQN sem filtro, DQN com filtro, Q-Learning tabular), **Etapa 4**
  (chaveamento reativo por campos potenciais) e as **três métricas** (tempo/frame, precisão de
  detecção, recompensa acumulada) foram implementadas conforme o Capítulo 3.
