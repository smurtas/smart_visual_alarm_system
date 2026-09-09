#include "camera.hpp"

#include "esp_log.h"


static const char *TAG = "CAMERA";

// ESP32-S3-EYE camera pins
static constexpr int CAM_PIN_XCLK  = 15;
static constexpr int CAM_PIN_SIOD  = 4;
static constexpr int CAM_PIN_SIOC  = 5;

static constexpr int CAM_PIN_D0    = 11;
static constexpr int CAM_PIN_D1    = 9;
static constexpr int CAM_PIN_D2    = 8;
static constexpr int CAM_PIN_D3    = 10;
static constexpr int CAM_PIN_D4    = 12;
static constexpr int CAM_PIN_D5    = 18;
static constexpr int CAM_PIN_D6    = 17;
static constexpr int CAM_PIN_D7    = 16;

static constexpr int CAM_PIN_VSYNC = 6;
static constexpr int CAM_PIN_HREF  = 7;
static constexpr int CAM_PIN_PCLK  = 13;


esp_err_t camera_init()
{
    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;

    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;

    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;

    config.pin_pwdn = -1;
    config.pin_reset = -1;

    config.xclk_freq_hz = 16000000;

    config.pixel_format = PIXFORMAT_JPEG;

    // QVGA keeps memory and processing requirements low.
    config.frame_size = FRAMESIZE_QVGA;  // 320 x 240
    config.jpeg_quality = 12;
    config.fb_count = 1;

    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    ESP_LOGI(TAG, "Initializing camera");

    const esp_err_t result = esp_camera_init(&config);

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Camera initialization failed: %s",
            esp_err_to_name(result)
        );
        return result;
    }

    sensor_t *sensor = esp_camera_sensor_get();

    if (sensor != nullptr) {
        // Required orientation for the ESP32-S3-EYE camera.
        sensor->set_vflip(sensor, 1);
        sensor->set_hmirror(sensor, 0);
    }

    ESP_LOGI(TAG, "Camera initialized successfully");

    return ESP_OK;
}


camera_fb_t *camera_capture()
{
    camera_fb_t *frame = esp_camera_fb_get();

    if (frame == nullptr) {
        ESP_LOGE(TAG, "Camera capture failed");
        return nullptr;
    }

    ESP_LOGI(
        TAG,
        "Frame captured: %u x %u, %u bytes, format=%d",
        static_cast<unsigned>(frame->width),
        static_cast<unsigned>(frame->height),
        static_cast<unsigned>(frame->len),
        static_cast<int>(frame->format)
    );

    return frame;
}


void camera_release(camera_fb_t *frame)
{
    if (frame != nullptr) {
        esp_camera_fb_return(frame);
    }
}