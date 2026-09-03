#pragma once

// #include "dl_model_base.hpp"
#include "esp_camera.h"
#include "esp_err.h"


struct Prediction {
    int class_index;
    const char *class_name;
    float confidence;
    float logits[3];
};


esp_err_t inference_init();

esp_err_t inference_run(
    const camera_fb_t *frame,
    Prediction *prediction
);

void inference_deinit();