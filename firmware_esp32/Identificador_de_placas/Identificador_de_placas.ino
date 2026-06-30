#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <TensorFlowLite_ESP32.h>
#include "img_converters.h"

// 1. Incluir o modelo convertido
#include "modelo_placas_grid_int8.h" // modelo retreinado COM negativas (Vitor, 29/06)

// 2. Incluir cabeçalhos do TensorFlow Lite Micro
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>

// Configurações de Rede e Telegram
// >>> PREENCHA com suas próprias credenciais antes de gravar na placa <<<
const char* ssid = "SUA_REDE_WIFI";
const char* password = "SUA_SENHA_WIFI";
#define BOT_TOKEN "SEU_TOKEN_DO_BOT_TELEGRAM"   // ex.: 123456789:AA...
#define CHAT_ID "SEU_CHAT_ID"                    // ex.: 123456789

WiFiClientSecure clientTCP;

// Definições do TensorFlow Lite
const int tensor_arena_size = 3 * 1024 * 1024; 
uint8_t* tensor_arena = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Definição de pinos padrão para a ESP32-S3 CAM
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5

#define Y2_GPIO_NUM 11
#define Y3_GPIO_NUM 9
#define Y4_GPIO_NUM 8
#define Y5_GPIO_NUM 10
#define Y6_GPIO_NUM 12
#define Y7_GPIO_NUM 18
#define Y8_GPIO_NUM 17
#define Y9_GPIO_NUM 16

#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

// --- VARIÁVEIS GLOBAIS ---
uint8_t* rgb_buffer = nullptr;

// --- CRONÓMETRO DE DISPARO (COOLDOWN) ---
unsigned long ultimo_disparo = 0;
// Defina aqui o tempo de pausa: 10000 = 10 segundos (para testes), 120000 = 2 minutos (para uso real)
const unsigned long COOLDOWN_TEMPO = 10000;

// --- FILTRO DE CONFIANÇA E ESTABILIDADE ---
// Só consideramos que existe mesmo uma placa se a confiança ficar ACIMA deste
// valor (antes era 0.50, o que gerava muitos disparos falsos)...
const float CONFIANCA_MINIMA = 0.95;
// ...e durante este número de frames SEGUIDOS. Isto evita disparar por causa de
// um único frame "sortudo". Aumente para ser mais rigoroso, diminua se ficar
// difícil demais detetar a placa.
const int FRAMES_NECESSARIOS = 2;
int frames_consecutivos = 0;

// Variáveis para a fragmentação (chunking) da foto do Telegram
uint8_t* telegram_photo_buf = nullptr;
size_t telegram_photo_len = 0;
size_t telegram_photo_index = 0;
size_t fatia_atual = 0;

bool isMoreDataAvailable() {
  if (telegram_photo_index >= telegram_photo_len) return false;
  size_t remaining = telegram_photo_len - telegram_photo_index;
  fatia_atual = (remaining > 512) ? 512 : remaining;
  return true;
}

uint8_t* getNextBuffer() {
  return telegram_photo_buf + telegram_photo_index;
}

int getBufferLength() {
  telegram_photo_index += fatia_atual;
  return fatia_atual;
}

void setup_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000; // OV5640 precisa de 20MHz para streamar (10MHz nao gerava frames)
    
    config.pixel_format = PIXFORMAT_JPEG; 
    config.frame_size = FRAMESIZE_QVGA; 
    config.jpeg_quality = 10;
    
    if(psramFound()){
      Serial.println("PSRAM Detetada! Alocando buffer de vídeo nos 8MB...");
      config.fb_location = CAMERA_FB_IN_PSRAM;
      config.fb_count = 2; 
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      Serial.println("AVISO: PSRAM desligada na IDE! Tentando alocar na RAM interna...");
      config.fb_location = CAMERA_FB_IN_DRAM;
      config.fb_count = 1;
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Erro ao inicializar a câmara OV5640");
        while (true);
    }
}

void setup_tflite() {
    tensor_arena = (uint8_t*)heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM);
    if (tensor_arena == nullptr) {
        Serial.println("Erro: Falha ao alocar tensor_arena na PSRAM!");
        while (true);
    }
    
    model = tflite::GetModel(modelo_placas_grid_int8);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Erro: Versão do esquema do modelo incompatível!");
        while (true);
    }

    static tflite::MicroErrorReporter error_reporter;
    static tflite::AllOpsResolver resolver;
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size, &error_reporter
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Erro ao alocar tensores!");
        while (true);
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    rgb_buffer = (uint8_t*)heap_caps_malloc(320 * 240 * 3, MALLOC_CAP_SPIRAM);
    if (rgb_buffer == nullptr) {
        Serial.println("Erro: Falha ao alocar buffer RGB na PSRAM!");
        while (true);
    }
}

void setup() {
    Serial.begin(115200);
    
    WiFi.begin(ssid, password);
    WiFi.setSleep(false); 
    
    clientTCP.setInsecure(); 
    clientTCP.setHandshakeTimeout(120000); 
    clientTCP.setTimeout(30000); 
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWi-Fi Ligado!");

    setup_camera();
    setup_tflite();
    
    Serial.println("Pipeline pronto para inferência.");
}

void loop() {
    // --- O TRUQUE DO DOUBLE DROP ---
    // Puxa o frame antigo da memória e devolve-o imediatamente (limpa o lag)
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
        esp_camera_fb_return(fb);
    }

    // Agora puxa um frame limpo, exatamente do momento atual
    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Falha ao capturar frame atual.");
        return;
    }

    // 1. Converte o JPEG da câmara para matriz RGB
    if (!fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_buffer)) {
        Serial.println("Erro ao descomprimir JPEG");
        esp_camera_fb_return(fb);
        return;
    }

    // 2. Parâmetros de Redimensionamento (320x240 -> 224x224)
    int target_w = 224;
    int target_h = 224;
    int src_w = 320;
    int src_h = 240;

    // 3. Parâmetros de Quantização do Modelo
    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;

    // 4. Redimensionar e Alimentar o Tensor
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_x = (x * src_w) / target_w;
            int src_y = (y * src_h) / target_h;
            
            int src_index = (src_y * src_w + src_x) * 3;
            int dst_index = (y * target_w + x) * 3;
            
            float r = rgb_buffer[src_index];
            float g = rgb_buffer[src_index + 1];
            float b = rgb_buffer[src_index + 2];
            
            float r_norm = (r / 127.5f) - 1.0f;
            float g_norm = (g / 127.5f) - 1.0f;
            float b_norm = (b / 127.5f) - 1.0f;

            input_tensor->data.int8[dst_index]     = (int8_t)(b_norm / input_scale + input_zero_point); // B
            input_tensor->data.int8[dst_index + 1] = (int8_t)(g_norm / input_scale + input_zero_point); // G
            input_tensor->data.int8[dst_index + 2] = (int8_t)(r_norm / input_scale + input_zero_point); // R
        }
    }
    
    // Executa a inferência
    if (interpreter->Invoke() == kTfLiteOk) {
        float maior_confianca = 0.0;
        int celula_da_placa = -1;

        for (int i = 0; i < 49; i++) {
            int index_confianca = i * 5; 
            int8_t valor_bruto = output_tensor->data.int8[index_confianca];
            float confianca_celula = (valor_bruto - output_tensor->params.zero_point) * output_tensor->params.scale;
            
            if (confianca_celula > maior_confianca) {
                maior_confianca = confianca_celula;
                celula_da_placa = i;
            }
        }

        Serial.print("Maior confianca encontrada na grelha: ");
        Serial.println(maior_confianca);

        // --- CONTAGEM DE FRAMES CONSECUTIVOS ---
        // Só consideramos uma deteção válida se a confiança se mantiver alta
        // durante vários frames seguidos. Um frame abaixo do limiar zera tudo.
        if (maior_confianca > CONFIANCA_MINIMA) {
            frames_consecutivos++;
        } else {
            frames_consecutivos = 0;
        }
        Serial.print("Frames consecutivos com placa: ");
        Serial.println(frames_consecutivos);

        if (frames_consecutivos >= FRAMES_NECESSARIOS) {

            // --- VERIFICAÇÃO DO COOLDOWN ---
            if (millis() - ultimo_disparo >= COOLDOWN_TEMPO || ultimo_disparo == 0) {
                
                Serial.println("Placa detetada! A desenhar a bounding box e ligar ao Telegram...");
                
                // --- 1. EXTRAIR A CAIXA E APLICAR A MATEMÁTICA YOLO ---
                int index_caixa = celula_da_placa * 5;
                
                float px = (output_tensor->data.int8[index_caixa + 1] - output_tensor->params.zero_point) * output_tensor->params.scale;
                float py = (output_tensor->data.int8[index_caixa + 2] - output_tensor->params.zero_point) * output_tensor->params.scale;
                float pw = (output_tensor->data.int8[index_caixa + 3] - output_tensor->params.zero_point) * output_tensor->params.scale;
                float ph = (output_tensor->data.int8[index_caixa + 4] - output_tensor->params.zero_point) * output_tensor->params.scale;

                int col = celula_da_placa % 7;
                int row = celula_da_placa / 7;

                int x_center = ((col + px) / 7.0) * 320;
                int y_center = ((row + py) / 7.0) * 240;
                int box_width = pw * 320;
                int box_height = ph * 240;

                int x_min = x_center - (box_width / 2);
                int y_min = y_center - (box_height / 2);
                int x_max = x_center + (box_width / 2);
                int y_max = y_center + (box_height / 2);

                if (x_min < 0) x_min = 0;
                if (y_min < 0) y_min = 0;
                if (x_max > 319) x_max = 319;
                if (y_max > 239) y_max = 239;

                // --- 2. DESENHAR O RETÂNGULO VERDE NO RGB_BUFFER ---
                int espessura = 3; 
                for (int y = y_min; y <= y_max; y++) {
                    for (int x = x_min; x <= x_max; x++) {
                        if (x < x_min + espessura || x > x_max - espessura || 
                            y < y_min + espessura || y > y_max - espessura) {
                            
                            int idx = (y * 320 + x) * 3;
                            rgb_buffer[idx] = 0;       // R
                            rgb_buffer[idx + 1] = 255; // G (Verde)
                            rgb_buffer[idx + 2] = 0;   // B
                        }
                    }
                }

                // --- 3. RECOMPRIMIR A IMAGEM PINTADA PARA JPEG ---
                Serial.println("A comprimir imagem modificada...");
                uint8_t* jpeg_buf = NULL;
                size_t jpeg_len = 0;
                
                bool convertido = fmt2jpg(rgb_buffer, 320 * 240 * 3, 320, 240, PIXFORMAT_RGB888, 30, &jpeg_buf, &jpeg_len);

                if (!convertido) {
                    Serial.println("Erro Crítico: Falha ao gerar o novo JPEG!");
                } else {
                    
                    // --- 4. ENVIO PARA O TELEGRAM ---
                    clientTCP.stop(); 
                    Serial.println("A ligar aos servidores do Telegram...");

                    if (clientTCP.connect("api.telegram.org", 443)) {
                        String BOUNDARY = "----ESP32Engenharia12345";
                        String head = "--" + BOUNDARY + "\r\n" +
                                      "Content-Disposition: form-data; name=\"chat_id\"\r\n\r\n" +
                                      String(CHAT_ID) + "\r\n" +
                                      "--" + BOUNDARY + "\r\n" +
                                      "Content-Disposition: form-data; name=\"photo\"; filename=\"img.jpg\"\r\n" +
                                      "Content-Type: image/jpeg\r\n\r\n";
                        String tail = "\r\n--" + BOUNDARY + "--\r\n";

                        uint32_t totalLen = head.length() + jpeg_len + tail.length();

                        clientTCP.println("POST /bot" + String(BOT_TOKEN) + "/sendPhoto HTTP/1.1");
                        clientTCP.println("Host: api.telegram.org");
                        clientTCP.println("Content-Length: " + String(totalLen));
                        clientTCP.println("Content-Type: multipart/form-data; boundary=" + BOUNDARY);
                        clientTCP.println();

                        clientTCP.print(head);

                        for (size_t n = 0; n < jpeg_len; n += 1024) {
                            size_t chunk = (jpeg_len - n > 1024) ? 1024 : (jpeg_len - n);
                            clientTCP.write(jpeg_buf + n, chunk);
                        }

                        clientTCP.print(tail);

                        long startTimer = millis();
                        boolean confirmacao = false;
                        while (clientTCP.connected() && (millis() - startTimer < 10000)) {
                            if (clientTCP.available()) {
                                String response = clientTCP.readStringUntil('\n');
                                if (response.indexOf("200 OK") != -1 || response.indexOf("\"ok\":true") != -1) {
                                    confirmacao = true;
                                }
                            }
                        }

                        if (confirmacao) {
                            Serial.println("🏆 SUCESSO! Foto com Bounding Box enviada!");
                            ultimo_disparo = millis(); // O ENVIO FUNCIONOU, RESETA O CRONÓMETRO AQUI!
                            frames_consecutivos = 0;   // zera a contagem para exigir nova sequência de frames
                        } else {
                            Serial.println("FALHA no envio do recibo.");
                        }
                        clientTCP.stop(); 
                    }
                    
                    // DESTRÓI O JPEG DA RAM APÓS O ENVIO
                    free(jpeg_buf); 
                }
                
                delay(1000); // Pequena pausa após processamento pesado
                
            } else {
                // Se detetou placa, mas ainda não passaram os 10 segundos
                Serial.println("IA viu uma placa, mas o Telegram está em arrefecimento (Cooldown).");
            }
        }
    }

    esp_camera_fb_return(fb);
    delay(10); 
}