# Firmware ESP32-S3 — Identificador de Placas

Código que **roda na placa** (Arduino). É o sistema embarcado: captura da câmera,
inferência com TensorFlow Lite Micro, desenho da *bounding box* e envio da foto ao
Telegram.

## Arquivos
```
Identificador_de_placas/
├── Identificador_de_placas.ino   ← código principal
├── modelo_placas_grid_int8.h     ← modelo INT8 (array C++, ~2,35 MB)
└── partitions.csv                ← tabela de partições (app de 5 MB)
```

## ⚠️ Credenciais (troque antes de publicar)
No início do `.ino` há credenciais do **autor** que **devem ser substituídas** pelas suas:
```cpp
const char* ssid     = "SUA_REDE";
const char* password = "SUA_SENHA";
#define BOT_TOKEN "SEU_TOKEN_DO_BOT_TELEGRAM"
#define CHAT_ID   "SEU_CHAT_ID"
```
> Não suba o `token` real do bot em repositório público — qualquer um poderia controlar o bot.

## Como gravar (Arduino IDE 2.x)
1. Instale o core **esp32 by Espressif** (3.3.8) em *Boards Manager*.
2. Abra a pasta `Identificador_de_placas/` (o `.ino` deve estar numa pasta de mesmo nome).
3. Selecione a placa **ESP32S3 Dev Module** e configure:
   - **PSRAM:** `OPI PSRAM`
   - **Flash Size:** `16MB`
   - **Partition Scheme:** `Custom` (usa o `partitions.csv` do sketch)
   - **USB CDC On Boot:** `Enabled` · **CPU Frequency:** `240 MHz`
4. Conecte a placa pela **porta USB nativa** (USB-Serial/JTAG) e clique em *Upload*.

> Esta placa tem **dois conectores USB**: grave pela porta **nativa** (entra em modo
> *download* sozinha). A outra porta (conversor UART) serve para ver a **saída serial**
> a 115200 baud, mas pode falhar no upload (*"Wrong boot mode"*).

### Alternativa: arduino-cli
```bash
FQBN="esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,PartitionScheme=custom,CDCOnBoot=default,CPUFreq=240"
arduino-cli compile --fqbn "$FQBN" Identificador_de_placas
arduino-cli upload  --fqbn "$FQBN" -p <PORTA_USB_NATIVA> Identificador_de_placas
```

## Observações de operação
- O `setup()` **espera o Wi-Fi conectar** antes de iniciar a câmera/modelo — garanta que a
  rede configurada esteja **ligada**, senão a placa fica parada no boot.
- A inferência embarcada leva **~2 minutos por frame** (modelo grande + *kernels* de
  referência do TFLite Micro + PSRAM). É o comportamento esperado.
- Disparo: confiança **≥ 0,95** por **2 frames** consecutivos → envia a foto ao Telegram.
