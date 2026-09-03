#include "wifi.hpp"
#include "secrets.hpp"

#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"


static const char *TAG = "WIFI";

static constexpr EventBits_t WIFI_CONNECTED_BIT = BIT0;
static constexpr EventBits_t WIFI_FAILED_BIT = BIT1;
static constexpr int MAX_RETRIES = 10;

static EventGroupHandle_t wifi_event_group = nullptr;

static int retry_count = 0;


static void wifi_event_handler(
    void *handler_argument,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data
)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Connecting to Wi-Fi");
        esp_wifi_connect();
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {

        if (retry_count < MAX_RETRIES) {
            ++retry_count;

            ESP_LOGW(
                TAG,
                "Connection failed, retry %d/%d",
                retry_count,
                MAX_RETRIES
            );

            esp_wifi_connect();
        } else {
            ESP_LOGE(TAG, "Maximum Wi-Fi retries reached");
            xEventGroupSetBits(wifi_event_group, WIFI_FAILED_BIT);
        }

        return;
    }

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        const auto *event =
            static_cast<ip_event_got_ip_t *>(event_data);

        ESP_LOGI(TAG, "Connected, IP: " IPSTR, IP2STR(&event->ip_info.ip));

        retry_count = 0;

        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}


esp_err_t wifi_init_sta()
{
    esp_err_t result = nvs_flash_init();

    if (result == ESP_ERR_NVS_NO_FREE_PAGES ||
        result == ESP_ERR_NVS_NEW_VERSION_FOUND) {

        ESP_ERROR_CHECK(nvs_flash_erase());
        result = nvs_flash_init();
    }

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "NVS initialization failed: %s", esp_err_to_name(result));
        return result;
    }

    wifi_event_group = xEventGroupCreate();

    if (wifi_event_group == nullptr) {
        ESP_LOGE(TAG, "Could not create Wi-Fi event group");
        return ESP_ERR_NO_MEM;
    }

    result = esp_netif_init();

    if (result != ESP_OK && result != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Network interface initialization failed: %s",
                 esp_err_to_name(result));
        return result;
    }

    result = esp_event_loop_create_default();

    if (result != ESP_OK && result != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Default event loop creation failed: %s",
                 esp_err_to_name(result));
        return result;
    }

    esp_netif_t *station_interface = esp_netif_create_default_wifi_sta();

    if (station_interface == nullptr) {
        ESP_LOGE(TAG, "Could not create Wi-Fi station interface");
        return ESP_FAIL;
    }

    wifi_init_config_t initialization = WIFI_INIT_CONFIG_DEFAULT();
    result = esp_wifi_init(&initialization);

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi driver initialization failed: %s",
                 esp_err_to_name(result));
        return result;
    }

    result = esp_event_handler_register(
        WIFI_EVENT,
        ESP_EVENT_ANY_ID,
        &wifi_event_handler,
        nullptr
    );

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi event handler registration failed: %s",
                 esp_err_to_name(result));
        return result;
    }

    result = esp_event_handler_register(
        IP_EVENT,
        IP_EVENT_STA_GOT_IP,
        &wifi_event_handler,
        nullptr
    );

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "IP event handler registration failed: %s",
                 esp_err_to_name(result));
        return result;
    }

    wifi_config_t configuration = {};

    std::strncpy(
        reinterpret_cast<char *>(configuration.sta.ssid),
        WIFI_SSID,
        sizeof(configuration.sta.ssid) - 1
    );

    std::strncpy(
        reinterpret_cast<char *>(configuration.sta.password),
        WIFI_PASSWORD,
        sizeof(configuration.sta.password) - 1
    );

    configuration.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    configuration.sta.pmf_cfg.capable = true;
    configuration.sta.pmf_cfg.required = false;

    result = esp_wifi_set_mode(WIFI_MODE_STA);

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Could not set Wi-Fi station mode: %s",
                 esp_err_to_name(result));
        return result;
    }

    result = esp_wifi_set_config(WIFI_IF_STA, &configuration);

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Could not apply Wi-Fi configuration: %s",
                 esp_err_to_name(result));
        return result;
    }

    result = esp_wifi_start();

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi start failed: %s", esp_err_to_name(result));
        return result;
    }

    const EventBits_t bits = xEventGroupWaitBits(
        wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAILED_BIT,
        pdFALSE,
        pdFALSE,
        portMAX_DELAY
    );

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Wi-Fi connection completed");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "Unable to connect to Wi-Fi");
    return ESP_FAIL;
}