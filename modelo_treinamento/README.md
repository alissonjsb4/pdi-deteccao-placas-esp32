# Treinamento e Exportação do Modelo

Notebook do **Google Colab** com todo o pipeline de Machine Learning: dataset,
arquitetura, treino (5 execuções), avaliação, **segmentação**, e exportação para
TensorFlow Lite / ESP32.

- 📓 **Notebook:** `Trabalho_Final_PDI_MobileNetV1_Grid_Detector.ipynb`
- 🔗 **Colab:** https://colab.research.google.com/drive/1DE-1luzYAd5wMEtShC5QgKr9YezZuzg7

## Conteúdo do notebook (seções)
1. Instalação e configuração (`IMG_SIZE=224`, `MobileNet α=0,5`).
2. Funções principais (decodificação grade→bbox, IoU, etc.).
3–4. Carregamento e combinação das bases (Base 1 do professor + Base 2 Brazil Plates).
5. **MobileNetV1 + detector por grade** (saída `7×7×5`).
6. *Data Augmentation*.
7–8. **Treinamento e seleção** do modelo (5 execuções, média e desvio).
9. Teste em imagens externas.
10. **Segmentação** dos caracteres da placa (OpenCV).
11. Avaliação em amostras.
12. **Exportação para TFLite** (Float32 / dynamic / INT8) e geração do `.h`.
13. Comparação Keras × TFLite (tempo, IoU, tamanho).

## Dataset
- **Base 1** (fornecida pelo professor) + **Base 2** *Brazil Plates Detector*
  (Roboflow, CC BY 4.0). Classe única `plate`, anotações YOLO.
- Divisão usada: Base 1 350/60, Base 2 200/12, mix ponderado 972/72, e **16 imagens
  negativas** (sem placa) para reduzir falsos positivos.

## Como retreinar / exportar
1. Abra o notebook no Colab (runtime com **GPU**).
2. Ajuste os caminhos do Google Drive nas primeiras células.
3. Execute as seções de treino (5 execuções) e selecione o melhor modelo.
4. A seção 12 exporta o `.tflite` INT8 e gera o cabeçalho C++:
   ```bash
   xxd -i modelo_grid_224_alpha_0p5_int8.tflite > modelo_placas_grid_int8.h
   ```
5. Copie o `.h` gerado para `firmware_esp32/Identificador_de_placas/` e o `.tflite`
   para `frontend/`.

## Hiperparâmetros (configuração final)
Adam · *fine-tuning* em 2 fases (12 + 50 épocas) · *batch* 8 · augmentation ·
`ReduceLROnPlateau` + `EarlyStopping` · entrada `224×224` · MobileNetV1 α=0,5 ·
quantização INT8 pós-treinamento.
