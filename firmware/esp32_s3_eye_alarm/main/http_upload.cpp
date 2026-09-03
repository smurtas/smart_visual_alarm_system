#include "http_upload.hpp"

#include "esp_http_client.h"
#include "esp_log.h"
#include "secrets.hpp"


static const char *TAG = "HTTP_UPLOAD";


static constexpr int HTTP_TIMEOUT_MS = 15000;


esp_err_t http_upload_frame(const camera_fb_t *frame)
{
    if (frame == nullptr || frame->buf == nullptr || frame->len == 0) {
        ESP_LOGE(TAG, "Invalid camera frame");
        return ESP_ERR_INVALID_ARG;
    }

    if (frame->format != PIXFORMAT_JPEG) {
        ESP_LOGE(
            TAG,
            "Frame is not JPEG; format=%d",
            static_cast<int>(frame->format)
        );
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(
        TAG,
        "Uploading JPEG: %u bytes",
        static_cast<unsigned>(frame->len)
    );

    esp_http_client_config_t configuration = {};

    configuration.url = UPLOAD_URL;
    configuration.method = HTTP_METHOD_POST;
    configuration.timeout_ms = HTTP_TIMEOUT_MS;
    configuration.keep_alive_enable = false;

    esp_http_client_handle_t client =
        esp_http_client_init(&configuration);

    if (client == nullptr) {
        ESP_LOGE(TAG, "HTTP client initialization failed");
        return ESP_ERR_NO_MEM;
    }

    esp_err_t result =
        esp_http_client_set_header(
            client,
            "Content-Type",
            "image/jpeg"
        );

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Could not set HTTP header: %s",
            esp_err_to_name(result)
        );

        esp_http_client_cleanup(client);
        return result;
    }

    result = esp_http_client_set_post_field(
        client,
        reinterpret_cast<const char *>(frame->buf),
        static_cast<int>(frame->len)
    );

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Could not set HTTP body: %s",
            esp_err_to_name(result)
        );

        esp_http_client_cleanup(client);
        return result;
    }

    result = esp_http_client_perform(client);

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "HTTP upload failed: %s",
            esp_err_to_name(result)
        );

        esp_http_client_cleanup(client);
        return result;
    }

    const int status_code = esp_http_client_get_status_code(client);
    const int64_t content_length =
        esp_http_client_get_content_length(client);

    ESP_LOGI(
        TAG,
        "HTTP response: status=%d, length=%lld",
        status_code,
        static_cast<long long>(content_length)
    );

    esp_http_client_cleanup(client);

    if (status_code != 201) {
        ESP_LOGE(TAG, "Unexpected HTTP status: %d", status_code);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "JPEG uploaded successfully");

    return ESP_OK;
}