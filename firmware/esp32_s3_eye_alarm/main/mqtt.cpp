#include "mqtt.hpp"

#include <cstdio>

#include "esp_log.h"
#include "mqtt_client.h"
#include "secrets.hpp"

static const char *TAG = "MQTT";


static constexpr const char *MQTT_TOPIC = "smartalarm/prediction";

static esp_mqtt_client_handle_t mqtt_client = nullptr;
static bool connected = false;


static void mqtt_event_handler(void *handler_args, esp_event_base_t event_base, int32_t event_id, void *event_data){
    const auto event = static_cast<esp_mqtt_event_handle_t>(event_data);

    switch (static_cast<esp_mqtt_event_id_t>(event_id)) {
        case MQTT_EVENT_CONNECTED:
            connected = true;

            ESP_LOGI(TAG,"Connected to broker");
            break;

        case MQTT_EVENT_DISCONNECTED:
            connected = false;

            ESP_LOGW(TAG,"Disconnected from broker");
            break;

        case MQTT_EVENT_PUBLISHED:
            ESP_LOGD( TAG, "Message published, id=%d", event->msg_id);
            break;

        case MQTT_EVENT_ERROR:
            connected = false;

            ESP_LOGE(TAG, "MQTT transport error");
            break;

        default:
            break;
    }
}


esp_err_t mqtt_init()
{
    if (mqtt_client != nullptr) {
        return ESP_OK;
    }

    ESP_LOGI(TAG,"Initializing MQTT client");
    esp_mqtt_client_config_t configuration = {};

    configuration.broker.address.uri = MQTT_BROKER_URI;
    configuration.credentials.client_id = "esp32-s3-eye-smart-alarm";
    configuration.session.keepalive = 60;

    mqtt_client = esp_mqtt_client_init(&configuration);

    if (mqtt_client == nullptr) {
        ESP_LOGE(TAG, "MQTT client allocation failed");

        return ESP_ERR_NO_MEM;
    }

    esp_err_t result =
        esp_mqtt_client_register_event(
            mqtt_client,
            MQTT_EVENT_ANY,
            mqtt_event_handler,
            nullptr
        );

    if (result != ESP_OK) {
        ESP_LOGE(TAG,"Event registration failed: %s", esp_err_to_name(result));

        esp_mqtt_client_destroy(mqtt_client);

        mqtt_client = nullptr;

        return result;
    }

    result =
        esp_mqtt_client_start(mqtt_client);

    if (result != ESP_OK) {
        ESP_LOGE( TAG, "MQTT client start failed: %s", esp_err_to_name(result));

        esp_mqtt_client_destroy(mqtt_client);

        mqtt_client = nullptr;

        return result;
    }

    ESP_LOGI(TAG,"MQTT client started");

    return ESP_OK;
}


bool mqtt_is_connected()
{
    return connected;
}


esp_err_t mqtt_publish_prediction(
    const char *class_name,
    float confidence
)
{
    if (
        mqtt_client == nullptr
        || !connected
    ) {
        ESP_LOGW(TAG,"Prediction not published: broker unavailable");

        return ESP_ERR_INVALID_STATE;
    }

    if (class_name == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    char payload[160];

    const int written = std::snprintf(
        payload,
        sizeof(payload),
        "{"
        "\"class\":\"%s\","
        "\"confidence\":%.4f"
        "}",
        class_name,
        static_cast<double>(confidence)
    );

    if (
        written < 0
        || written >= static_cast<int>(
            sizeof(payload)
        )
    ) {
        ESP_LOGE(TAG,"MQTT payload generation failed");

        return ESP_ERR_INVALID_SIZE;
    }

    const int message_id =
        esp_mqtt_client_publish(
            mqtt_client,
            MQTT_TOPIC,
            payload,
            0,      // automatic payload length
            0,      // QoS 0
            0       // retain false
        );

    if (message_id < 0) {
        ESP_LOGE(TAG, "Prediction publish failed");

        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Published: %s",payload);

    return ESP_OK;
}