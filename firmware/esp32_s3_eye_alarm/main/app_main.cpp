#include "camera.hpp"
#include "http_upload.hpp"
#include "inference.hpp"
#include "mqtt.hpp"
#include "wifi.hpp"

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"


static const char *TAG = "SMART_ALARM";

static constexpr float ALERT_THRESHOLD = 0.85f;
static constexpr int INFERENCE_INTERVAL_MS = 2000;


extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting Smart Visual Alarm");

    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed");
        return;
    }

    if (inference_init() != ESP_OK) {
        ESP_LOGE(TAG, "Inference initialization failed");
        return;
    }

    if (wifi_init_sta() != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi initialization failed");
        return;
    }

    if (mqtt_init() != ESP_OK) {
        ESP_LOGE(TAG, "MQTT initialization failed");
        return;
    }

    ESP_LOGI(TAG, "Camera warm-up started");

    for (int index = 0; index < 8; ++index) {
        camera_fb_t *warmup_frame = camera_capture();

        if (warmup_frame != nullptr) {
            camera_release(warmup_frame);
        }

        vTaskDelay(pdMS_TO_TICKS(200));
    }

    ESP_LOGI(TAG, "System ready");

    while (true) {
        camera_fb_t *frame = camera_capture();

        if (frame == nullptr) {
            ESP_LOGE(TAG, "Frame acquisition failed");

            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        Prediction prediction = {};

        const esp_err_t inference_result =
            inference_run(frame, &prediction);

        if (inference_result == ESP_OK) {
            ESP_LOGI(
                TAG,
                "Prediction: %s | confidence: %.2f%%",
                prediction.class_name,
                prediction.confidence * 100.0f
            );

            ESP_LOGI(
                TAG,
                "Logits: animal=%.4f empty=%.4f person=%.4f",
                prediction.logits[0],
                prediction.logits[1],
                prediction.logits[2]
            );

            const bool is_alert_class =
                prediction.class_index == 0
                || prediction.class_index == 2;

            const bool is_alert =
                is_alert_class
                && prediction.confidence >= ALERT_THRESHOLD;

            if (is_alert) {
                const esp_err_t upload_result =
                    http_upload_frame(frame);

                if (upload_result != ESP_OK) {
                    ESP_LOGE(
                        TAG,
                        "Alert image upload failed: %s",
                        esp_err_to_name(upload_result)
                    );
                }

                if (mqtt_is_connected()) {
                    const esp_err_t mqtt_result =
                        mqtt_publish_prediction(
                            prediction.class_name,
                            prediction.confidence
                        );

                    if (mqtt_result != ESP_OK) {
                        ESP_LOGE(
                            TAG,
                            "MQTT publish failed: %s",
                            esp_err_to_name(mqtt_result)
                        );
                    }
                } else {
                    ESP_LOGW(TAG, "MQTT not connected yet");
                }
            }
            else if (is_alert_class) {
                ESP_LOGI(
                    TAG,
                    "Alert suppressed: confidence %.2f%% below threshold %.2f%%",
                    prediction.confidence * 100.0f,
                    ALERT_THRESHOLD * 100.0f
                );
            }
            else if (mqtt_is_connected()) {
                const esp_err_t mqtt_result =
                    mqtt_publish_prediction(
                        prediction.class_name,
                        prediction.confidence
                    );

                if (mqtt_result != ESP_OK) {
                    ESP_LOGE(
                        TAG,
                        "MQTT publish failed: %s",
                        esp_err_to_name(mqtt_result)
                    );
                }
            }
        } else {
            ESP_LOGE(
                TAG,
                "Inference failed: %s",
                esp_err_to_name(inference_result)
            );
        }

        camera_release(frame);

        vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
    }
}