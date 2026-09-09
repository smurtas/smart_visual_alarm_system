#pragma once

#include "esp_camera.h"
#include "esp_err.h"

esp_err_t http_upload_frame(const camera_fb_t *frame);