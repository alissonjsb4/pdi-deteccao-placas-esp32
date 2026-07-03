# Kit de slides — texto pronto para colar no Canva

Deck atual: 15 slides ("Detecção de Placas com MobileNetV1 e Grid Detector").
Este kit adiciona **8 slides novos** (inserir depois do slide 12 — "Deploy em
Sistemas Embarcados") e lista **2 correções** no deck existente.
Os PNGs citados estão nesta pasta e já vêm no fundo navy do deck (16:9).

Legenda: 🖼 = asset pronto · 🗣 = nota do orador (não vai no slide).

---

## Correções no deck existente

1. **Slide 8 (Detector por Grade)** — na coluna "Resultado", o texto
   "(confiança, cx, cy, w, h)" está **sobreposto** ao parágrafo "Esse resultado
   é usado…". Separar em duas linhas: título pequeno "(confiança, cx, cy, w, h)"
   e o parágrafo abaixo.
2. **Slide 15 ("Links dos datasets")** — o conteúdo é crédito de *imagens de
   ilustração*, não de datasets. Renomear para **"Créditos das imagens"** e
   acrescentar um bloco "Datasets" com os links reais:
   - Brazil Plates Detector (Roboflow, CC BY 4.0):
     https://universe.roboflow.com/testes-bbb7m/brazil-plates-detector-zn2t0
   - Base do professor (TI0147) — uso interno da disciplina.
   - Repositório do trabalho: github.com/<usuario>/<repo> *(preencher)*

Números conferidos slides ↔ relatório ↔ repositório (nada a corrigir):
IoU 0,731 · Recall 0,922 · F1 0,944 · PC 4–7 ms · ESP 117 s · 12+50 épocas ·
batch 8 · negativas 12 treino + 4 validação · modelo 2,35 MB INT8.

---

## Slide N1 — Firmware na ESP32-S3: como o código funciona

🖼 `fig_pipeline_firmware.png` (imagem única, ocupa o slide)

Bullets (se quiser texto além da figura):
- Um único sketch Arduino (`Identificador_de_placas.ino`): `setup_camera()` +
  `setup_tflite()` + `loop()`.
- Modelo INT8 (2,35 MB) embutido no binário como header C++; arena de tensores
  de 3 MB alocada na PSRAM de 8 MB; partição de flash customizada (5 MB de app).
- Loop: captura JPEG → converte RGB → redimensiona/quantiza → `Invoke()` →
  decodifica a grade 7×7 → filtro de estabilidade → desenha a caixa → envia ao
  Telegram.

🗣 "O firmware é autossuficiente: da captura ao Telegram, tudo acontece dentro
do microcontrolador — o PC não participa."

---

## Slide N2 — Por que ~2 minutos por frame (e por que tudo bem)

🖼 `fig_tempo_log.png`

Bullets:
- O MESMO arquivo `.tflite` roda em 4–7 ms no PC e ~117 s na ESP32-S3
  (~17.000×): o gargalo é a plataforma, não o modelo.
- Três causas: kernels *de referência* da TFLite Micro (a lib usada não tem os
  kernels ESP-NN otimizados do S3), pesos/ativações na PSRAM (mais lenta que a
  SRAM interna) e entrada 224×224 (grande para 240 MHz).
- Caminhos de otimização mapeados: migrar para `esp-tflite-micro` com ESP-NN
  (ganho típico de 5–10×) e/ou retreinar com entrada 96×96 (custo cai ~5×).
- Decisão do projeto: manter 224×224 e a lib estável = prova de conceito de
  edge AI 100% offline, com precisão preservada.

🗣 "Não é um bug, é o custo real de rodar uma CNN em um microcontrolador sem
aceleração — e é exatamente essa medição que o trabalho entrega."

---

## Slide N3 — Operação em campo: filtro de estabilidade ao vivo

🖼 `fig_confianca_campo.png` (dados reais de 02/07)

Bullets:
- Regra de disparo: confiança ≥ limiar por **2 frames consecutivos** + cooldown
  de 10 s → elimina disparos por um único frame "sortudo".
- O limiar é **calibrável em campo** (0,85–0,95 conforme iluminação/cena); na
  sessão do gráfico foi recalibrado 0,95 → 0,90 → 0,85 e o sistema disparou.
- Da detecção à foto no Telegram: **+18 s** medidos.
- Faixa sem placa medida: 0,42–0,58 → margem grande para o limiar (sem falsos
  positivos).

🗣 "Cada ponto é uma inferência de ~2 min. Os dois pontos verdes finais são os
2 frames consecutivos acima do limiar; 18 segundos depois a foto chegou no
Telegram do grupo."

---

## Slide N4 — Alternativa considerada: inferência na nuvem

(sem asset — slide de texto/duas colunas)

Coluna A — "ESP só captura, servidor infere":
- Latência total de ~2 min cai para ~1–2 s.
- Mas exige conectividade estável, um servidor sempre ligado (custo) e envia
  imagens da rua para fora do dispositivo (privacidade).

Coluna B — "Nossa escolha: inferência embarcada":
- 100% offline após o boot (Wi-Fi só para notificar), sem servidor, sem custo
  recorrente, dado bruto nunca sai do dispositivo.
- Robustez: o detector continua funcionando se a internet cair (a foto é o
  único passo que depende de rede).
- É o cenário-alvo do enunciado: modelo embarcado no microcontrolador.

🗣 "A nuvem resolveria a latência, mas descaracterizaria o trabalho: viraria
um streamer de câmera. A versão embarcada é mais lenta, porém autônoma — e a
comparação PC×ESP do slide anterior quantifica exatamente esse trade-off."

---

## Slide N5 — Frontend de apresentação (visão geral)

(asset opcional: print do navegador em http://localhost:8000)

Bullets:
- App web local e offline (Python + LiteRT + OpenCV): `python
  frontend/frontend_apresentacao.py` → http://localhost:8000.
- Roda o MESMO `.tflite` embarcado, no PC (4–7 ms) — base da comparação de
  plataformas.
- 4 abas = as três modalidades do enunciado + métricas: **Imagem**, **Vídeo**,
  **Tempo real** (ESP via serial) e **Métricas**.

🗣 "É a mesma rede da ESP; só muda onde ela executa. Tudo que o professor pedir
para ver, dá pra demonstrar ao vivo nessa tela."

---

## Slide N6 — Frontend: aba Imagem

🖼 `aba_imagem_1.png` (há também `aba_imagem_2/3.png` de reserva)

Bullets:
- Upload de qualquer foto → detecção com bounding box + confiança.
- Mapa de confiança da grade 7×7: mostra *onde* a rede "acendeu".
- Segmentação dos caracteres (OpenCV: CLAHE → BlackHat → Otsu → componentes
  conectados) na resolução original.

🗣 "O heatmap é a saída bruta da rede; a célula vermelha é a que decide a
caixa. A segmentação é pós-processamento clássico de PDI, sem rede neural."

---

## Slide N7 — Frontend: aba Vídeo

🖼 `aba_video.png`

Bullets:
- Detecção quadro a quadro com anotação ao vivo (stream MJPEG no navegador).
- Confiança de até 0,996 no melhor frame do vídeo de teste; centenas de FPS
  possíveis no PC.
- Mesmo pipeline da aba Imagem, aplicado continuamente.

🗣 "No PC a rede é tão rápida que o gargalo vira o próprio vídeo — por isso a
comparação com os 117 s da ESP é tão ilustrativa."

---

## Slide N8 — Frontend: aba Tempo real (a ESP ao vivo)

🖼 reaproveita `fig_confianca_campo.png` OU um print ao vivo do dashboard

Bullets:
- Conecta na ESP pela serial (COM3) e reconstrói o estado em tempo real:
  gráfico confiança×tempo, barra de progresso da inferência (~117 s), log de
  eventos (boot, Wi-Fi, detecção, envio).
- Quando a ESP dispara, a foto com bounding box chega no Telegram — o dashboard
  registra o instante.
- Aba Métricas: as tabelas do relatório (5 execuções, PC×ESP) para consulta.

🗣 "Se der tempo na demo: conectar ao vivo e mostrar uma inferência acontecendo
com a barra de progresso — e a foto caindo no Telegram."

---

## Slide N9 — Saída final: a mensagem no Telegram

🖼 **usar um print do celular** com o chat do bot (tem fotos reais das sessões
de 30/06 e 02/07) — de preferência mostrando a foto anotada + o horário.

Bullets:
- Confirmada a detecção (2 frames ≥ limiar), a ESP desenha a bounding box
  verde na imagem, recomprime para JPEG e envia via **API do Telegram**
  (POST multipart/form-data direto no `api.telegram.org`, sem servidor
  intermediário).
- A foto chega no chat **~18 s** após a detecção (medido em campo) — upload em
  chunks de 1 KB sobre TLS.
- **Cooldown de 10 s** após cada envio evita spam com o carro parado na frente
  da câmera; a contagem de frames zera e o ciclo recomeça.
- Robustez: timeouts de 15 s derrubam conexões penduradas; se o envio falhar,
  o sistema continua detectando e tenta no próximo disparo.

🗣 "O Telegram é o 'atuador' do sistema: a prova de que o pipeline embarcado
fechou o ciclo — da luz que entrou na lente até a notificação no celular —
sem nenhum computador no meio."

---

## Ordem sugerida do deck final (24 slides)

1–12 = atuais · **N1, N2, N3, N4** (bloco embarcado) · **N9** (Telegram) ·
**N5, N6, N7, N8** (bloco frontend) · 13 (Resultados) · 14 (Conclusão) ·
15 (Créditos corrigido).

Dica: no slide 13 ("Análise de Resultados em Campo"), citar oralmente que a
tabela completa com desvios está no relatório (5 execuções).

Obs.: o PDF do deck não é versionado no repo enquanto estiver em edição
(`apresentacao/*.pdf` está no .gitignore); commitar apenas a versão final.
