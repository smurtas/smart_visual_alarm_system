#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char* WIFI_SSID = "inserisci rete";
const char* WIFI_PASSWORD = "cambia password";
const char* SERVER_URL =
    "http://192.168.1.129:5000/upload?label=empty";

const unsigned long CAPTURE_INTERVAL_MS = 2000;
unsigned long lastCaptureTime = 0;

// Pin ufficiali ESP32-S3-EYE
#define CAMERA_PIN_PWDN   -1
#define CAMERA_PIN_RESET  -1
#define CAMERA_PIN_XCLK   15
#define CAMERA_PIN_SIOD   4
#define CAMERA_PIN_SIOC   5

#define CAMERA_PIN_D0     11
#define CAMERA_PIN_D1     9
#define CAMERA_PIN_D2     8
#define CAMERA_PIN_D3     10
#define CAMERA_PIN_D4     12
#define CAMERA_PIN_D5     18
#define CAMERA_PIN_D6     17
#define CAMERA_PIN_D7     16

#define CAMERA_PIN_VSYNC  6
#define CAMERA_PIN_HREF   7
#define CAMERA_PIN_PCLK   13


bool initCamera() {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = CAMERA_PIN_D0;
  config.pin_d1 = CAMERA_PIN_D1;
  config.pin_d2 = CAMERA_PIN_D2;
  config.pin_d3 = CAMERA_PIN_D3;
  config.pin_d4 = CAMERA_PIN_D4;
  config.pin_d5 = CAMERA_PIN_D5;
  config.pin_d6 = CAMERA_PIN_D6;
  config.pin_d7 = CAMERA_PIN_D7;

  config.pin_xclk = CAMERA_PIN_XCLK;
  config.pin_pclk = CAMERA_PIN_PCLK;
  config.pin_vsync = CAMERA_PIN_VSYNC;
  config.pin_href = CAMERA_PIN_HREF;
  config.pin_sccb_sda = CAMERA_PIN_SIOD;
  config.pin_sccb_scl = CAMERA_PIN_SIOC;
  config.pin_pwdn = CAMERA_PIN_PWDN;
  config.pin_reset = CAMERA_PIN_RESET;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Per la diagnosi usiamo un solo framebuffer.
  if (psramFound()) {
    Serial.println("PSRAM rilevata");

    config.frame_size = FRAMESIZE_VGA;       // 640 x 480
    config.jpeg_quality = 10;                // Numero più basso = qualità maggiore
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("PSRAM non rilevata");

    config.frame_size = FRAMESIZE_QVGA;      // 320 x 240
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("Errore camera: 0x%x\n", err);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor == nullptr) {
    Serial.println("Errore: sensore non disponibile");
    return false;
  }

  Serial.printf("PID sensore: 0x%02X\n", sensor->id.PID);

  // Configurazione usata dall'esempio ufficiale per ESP32-S3-EYE.
  sensor->set_vflip(sensor, 1);

  // Manteniamo attivi esposizione, guadagno e bilanciamento automatici.
  sensor->set_whitebal(sensor, 1);
  sensor->set_awb_gain(sensor, 1);
  sensor->set_exposure_ctrl(sensor, 1);
  sensor->set_gain_ctrl(sensor, 1);

  sensor->set_brightness(sensor, 0);
  sensor->set_contrast(sensor, 0);
  sensor->set_saturation(sensor, 0);

  Serial.println("Camera inizializzata");
  return true;
}


bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(1000);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connessione Wi-Fi");

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    if (millis() - start > 30000) {
      Serial.println();
      Serial.println("Timeout Wi-Fi");
      Serial.printf("Stato Wi-Fi: %d\n", WiFi.status());
      return false;
    }
  }

  Serial.println();
  Serial.println("Wi-Fi connesso");
  Serial.print("IP ESP32: ");
  Serial.println(WiFi.localIP());

  return true;
}


void warmUpCamera() {
  Serial.println("Stabilizzazione esposizione camera...");
  delay(3000);

  // I primi frame possono essere neri o fortemente sottoesposti.
  for (int i = 0; i < 8; i++) {
    camera_fb_t* frame = esp_camera_fb_get();

    if (frame != nullptr) {
      Serial.printf(
        "Frame %d scartato: %u byte, %ux%u\n",
        i + 1,
        frame->len,
        frame->width,
        frame->height
      );

      esp_camera_fb_return(frame);
    } else {
      Serial.printf("Frame %d non acquisito\n", i + 1);
    }

    delay(300);
  }
}


bool captureAndUpload() {
  camera_fb_t* frame = esp_camera_fb_get();

  if (frame == nullptr) {
    Serial.println("Acquisizione fallita");
    return false;
  }

  Serial.printf(
    "Foto finale: %u byte, %ux%u, formato %d\n",
    frame->len,
    frame->width,
    frame->height,
    frame->format
  );

  // Un JPEG VGA di pochi byte è quasi certamente anomalo.
  if (frame->len < 1000) {
    Serial.println("Errore: frame troppo piccolo");
    esp_camera_fb_return(frame);
    return false;
  }

  WiFiClient client;
  HTTPClient http;

  http.setConnectTimeout(10000);
  http.setTimeout(15000);

  if (!http.begin(client, SERVER_URL)) {
    Serial.println("Inizializzazione HTTP fallita");
    esp_camera_fb_return(frame);
    return false;
  }

  http.addHeader("Content-Type", "image/jpeg");

  int responseCode = http.POST(frame->buf, frame->len);

  Serial.printf("Codice HTTP: %d\n", responseCode);

  if (responseCode > 0) {
    Serial.println(http.getString());
  } else {
    Serial.println(http.errorToString(responseCode));
  }

  http.end();
  esp_camera_fb_return(frame);

  return responseCode == 201;
}


void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("Avvio ESP32-S3-EYE");

  if (!initCamera()) {
    return;
  }

  if (!connectWiFi()) {
    return;
  }

  warmUpCamera();
  Serial.println("Avvio raccolta automatica ogni 500 ms");
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
    lastCaptureTime = currentTime;

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("Wi-Fi disconnesso, tentativo di riconnessione...");

      if (!connectWiFi()) {
        Serial.println("Riconnessione fallita");
        delay(1000);
        return;
      }
    }

    bool success = captureAndUpload();

    if (!success) {
      Serial.println("Foto non inviata");
    }
  }

  delay(10);
}