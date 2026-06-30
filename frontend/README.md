# Frontend de Apresentação

Aplicação **web local** (offline) que demonstra as **três modalidades** exigidas
(imagem, vídeo e tempo real) usando o **mesmo modelo** embarcado na ESP, mas rodando no
PC via **LiteRT** (~4 ms/inferência). A segmentação usa **OpenCV**.

## Dependências
```bash
pip install ai-edge-litert opencv-python pillow pyserial numpy
```
> `ai-edge-litert` é o runtime do TensorFlow Lite (funciona inclusive no Python 3.14).

## Como rodar
```bash
python frontend_apresentacao.py          # porta padrão 8000
python frontend_apresentacao.py 9000     # outra porta web
```
Depois abra **http://localhost:8000** no navegador.

## Abas
| Aba | O que faz |
|---|---|
| **Imagem** | Escolha/arraste uma imagem → detecção (bbox + mapa de confiança da grade) + segmentação dos caracteres. Modos RGB e *gray3*. |
| **Vídeo** | Seleciona um vídeo de `videos_teste/` → detecção quadro a quadro ao vivo (MJPEG) com FPS. |
| **Tempo real** | Conecta na **ESP** pela porta serial (ex.: `COM3`) e mostra a confiança ao vivo (gráfico tempo × confiança), barra de progresso da inferência e o log de eventos. |
| **Métricas** | Tabelas com os resultados reais (5 execuções, PC × ESP, efeito das negativas) — ótimas para prints. |

## Observações
- A aba **Tempo real** abre a porta serial e **reinicia a ESP** ao conectar; o primeiro
  frame leva ~2 min (inferência embarcada). Garanta o Wi-Fi da ESP ligado.
- O modelo usado é `modelo_grid_224_alpha_0p5_int8.tflite` (nesta mesma pasta) — o mesmo
  que foi embarcado na placa.
- Tudo é servido localmente, **sem internet**, para funcionar na hora da apresentação.
