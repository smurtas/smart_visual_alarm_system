#pragma once

#include "esp_err.h"

esp_err_t mqtt_init();
bool mqtt_is_connected();
esp_err_t mqtt_publish_prediction(const char *class_name, float confidence);