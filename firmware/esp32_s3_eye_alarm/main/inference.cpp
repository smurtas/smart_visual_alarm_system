#include "inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "dl_math.hpp"

#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "img_converters.h"


static const char *TAG = "INFERENCE";

static constexpr int MODEL_WIDTH = 48;
static constexpr int MODEL_HEIGHT = 48;
static constexpr int MODEL_CHANNELS = 3;

static constexpr int CAMERA_WIDTH = 320;
static constexpr int CAMERA_HEIGHT = 240;

static constexpr const char *CLASS_NAMES[3] = {
    "animal",
    "empty",
    "person"
};


// Normalizzazione usata durante training e quantizzazione.
static constexpr float IMAGENET_MEAN[3] = {
    0.485f,
    0.456f,
    0.406f
};

static constexpr float IMAGENET_STD[3] = {
    0.229f,
    0.224f,
    0.225f
};


extern const uint8_t mcunet_int8_espdl[]
    asm("_binary_mcunet_int8_espdl_start");


static dl::Model *model = nullptr;
static dl::TensorBase *model_input = nullptr;
static dl::TensorBase *model_output = nullptr;


/**
 * Convert the camera JPEG to RGB888, resize it to 48 x 48,
 * normalize the pixels and quantize them to INT8.
 *
 * The ESP-DL model expects NHWC input:
 *
 * [1, 48, 48, 3]
 */
static esp_err_t preprocess_frame(
    const camera_fb_t *frame,
    dl::TensorBase *input
)
{
    if (frame == nullptr || input == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    if (
        frame->width != CAMERA_WIDTH
        || frame->height != CAMERA_HEIGHT
    ) {
        ESP_LOGE(
            TAG,
            "Unexpected frame size: %ux%u",
            static_cast<unsigned>(frame->width),
            static_cast<unsigned>(frame->height)
        );

        return ESP_ERR_INVALID_SIZE;
    }

    const size_t rgb_size =
        CAMERA_WIDTH
        * CAMERA_HEIGHT
        * MODEL_CHANNELS;

    uint8_t *rgb888 = static_cast<uint8_t *>(
        heap_caps_malloc(
            rgb_size,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        )
    );

    if (rgb888 == nullptr) {
        ESP_LOGE(TAG, "Could not allocate RGB888 buffer");
        return ESP_ERR_NO_MEM;
    }

    const bool converted = fmt2rgb888(
        frame->buf,
        frame->len,
        frame->format,
        rgb888
    );

    if (!converted) {
        ESP_LOGE(TAG, "JPEG to RGB888 conversion failed");
        heap_caps_free(rgb888);
        return ESP_FAIL;
    }

    int8_t *input_data =
        input->get_element_ptr<int8_t>();

    if (input_data == nullptr) {
        ESP_LOGE(TAG, "Model input buffer is null");
        heap_caps_free(rgb888);
        return ESP_FAIL;
    }

    const float inverse_scale =
        DL_RESCALE(input->exponent);

    int minimum_value = 127;
    int maximum_value = -128;
    long input_sum = 0;

    /*
 * Bilinear resize:
 *
 * 320 x 240 -> 48 x 48
 *
 * The destination layout is NHWC:
 * R, G, B, R, G, B, ...
 */
for (int target_y = 0; target_y < MODEL_HEIGHT; ++target_y) {

    const float source_y =
        (target_y + 0.5f)
        * CAMERA_HEIGHT
        / MODEL_HEIGHT
        - 0.5f;

    int y0 = static_cast<int>(std::floor(source_y));
    int y1 = y0 + 1;

    const float wy = source_y - std::floor(source_y);

    y0 = std::max(0, std::min(y0, CAMERA_HEIGHT - 1));
    y1 = std::max(0, std::min(y1, CAMERA_HEIGHT - 1));

    for (int target_x = 0; target_x < MODEL_WIDTH; ++target_x) {

        const float source_x =
            (target_x + 0.5f)
            * CAMERA_WIDTH
            / MODEL_WIDTH
            - 0.5f;

        int x0 = static_cast<int>(std::floor(source_x));
        int x1 = x0 + 1;

        const float wx = source_x - std::floor(source_x);

        x0 = std::max(0, std::min(x0, CAMERA_WIDTH - 1));
        x1 = std::max(0, std::min(x1, CAMERA_WIDTH - 1));

        const int index00 =
            (y0 * CAMERA_WIDTH + x0) * MODEL_CHANNELS;

        const int index01 =
            (y0 * CAMERA_WIDTH + x1) * MODEL_CHANNELS;

        const int index10 =
            (y1 * CAMERA_WIDTH + x0) * MODEL_CHANNELS;

        const int index11 =
            (y1 * CAMERA_WIDTH + x1) * MODEL_CHANNELS;

        const int target_index =
            (target_y * MODEL_WIDTH + target_x)
            * MODEL_CHANNELS;

        for (int channel = 0; channel < MODEL_CHANNELS; ++channel) {

            const float top =
                rgb888[index00 + channel] * (1.0f - wx)
                + rgb888[index01 + channel] * wx;

            const float bottom =
                rgb888[index10 + channel] * (1.0f - wx)
                + rgb888[index11 + channel] * wx;

            const float resized_pixel =
                top * (1.0f - wy)
                + bottom * wy;

            const float pixel = resized_pixel / 255.0f;

            const float normalized =
                (pixel - IMAGENET_MEAN[channel])
                / IMAGENET_STD[channel];

            const int8_t quantized_value =
                dl::quantize<int8_t>(
                    normalized,
                    inverse_scale
                );

            input_data[target_index + channel] =
                quantized_value;

            minimum_value = std::min(
                minimum_value,
                static_cast<int>(quantized_value)
            );

            maximum_value = std::max(
                maximum_value,
                static_cast<int>(quantized_value)
            );

            input_sum += quantized_value;
        }
    }
}

    const int input_elements =
        MODEL_WIDTH
        * MODEL_HEIGHT
        * MODEL_CHANNELS;

    const float input_average =
        static_cast<float>(input_sum)
        / static_cast<float>(input_elements);

    ESP_LOGI(
        TAG,
        "Input tensor statistics: min=%d max=%d mean=%.2f",
        minimum_value,
        maximum_value,
        input_average
    );

    heap_caps_free(rgb888);

    return ESP_OK;
}


static void calculate_softmax(
    const float logits[3],
    float probabilities[3]
)
{
    const float maximum = std::max(
        logits[0],
        std::max(
            logits[1],
            logits[2]
        )
    );

    float denominator = 0.0f;

    for (int index = 0; index < 3; ++index) {
        probabilities[index] =
            std::exp(
                logits[index] - maximum
            );

        denominator += probabilities[index];
    }

    if (denominator <= 0.0f) {
        probabilities[0] = 0.0f;
        probabilities[1] = 0.0f;
        probabilities[2] = 0.0f;
        return;
    }

    for (int index = 0; index < 3; ++index) {
        probabilities[index] /= denominator;
    }
}


esp_err_t inference_init()
{
    if (model != nullptr) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Loading MCUNet model");

    model = new dl::Model(
        reinterpret_cast<const char *>(
            mcunet_int8_espdl
        ),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );

    if (model == nullptr) {
        ESP_LOGE(TAG, "Model allocation failed");
        return ESP_ERR_NO_MEM;
    }

    model_input = model->get_input();
    model_output = model->get_output();

    if (
        model_input == nullptr
        || model_output == nullptr
    ) {
        ESP_LOGE(
            TAG,
            "Could not obtain model tensors"
        );

        inference_deinit();

        return ESP_FAIL;
    }

    ESP_LOGI(
        TAG,
        "Input tensor: size=%d exponent=%d",
        model_input->get_size(),
        static_cast<int>(model_input->exponent)
    );

    ESP_LOGI(
        TAG,
        "Output tensor: size=%d exponent=%d",
        model_output->get_size(),
        static_cast<int>(model_output->exponent)
    );

    const int expected_input_size =
        MODEL_WIDTH
        * MODEL_HEIGHT
        * MODEL_CHANNELS;

    if (
        model_input->get_size()
        != expected_input_size
    ) {
        ESP_LOGE(
            TAG,
            "Unexpected model input size: %d; expected: %d",
            model_input->get_size(),
            expected_input_size
        );

        inference_deinit();

        return ESP_ERR_INVALID_SIZE;
    }

    if (model_output->get_size() != 3) {
        ESP_LOGE(
            TAG,
            "Unexpected model output size: %d",
            model_output->get_size()
        );

        inference_deinit();

        return ESP_ERR_INVALID_SIZE;
    }

    ESP_LOGI(TAG, "MCUNet model ready");

    return ESP_OK;
}


esp_err_t inference_run(
    const camera_fb_t *frame,
    Prediction *prediction
)
{
    if (
        model == nullptr
        || model_input == nullptr
        || model_output == nullptr
        || prediction == nullptr
    ) {
        return ESP_ERR_INVALID_STATE;
    }

    const esp_err_t preprocess_result =
        preprocess_frame(
            frame,
            model_input
        );

    if (preprocess_result != ESP_OK) {
        return preprocess_result;
    }

    model->run();

    const int8_t *output_data =
        model_output->get_element_ptr<int8_t>();

    if (output_data == nullptr) {
        ESP_LOGE(
            TAG,
            "Model output buffer is null"
        );

        return ESP_FAIL;
    }

    const float output_scale =
        DL_SCALE(model_output->exponent);

    for (int index = 0; index < 3; ++index) {
        prediction->logits[index] =
            dl::dequantize(
                output_data[index],
                output_scale
            );
    }

    float probabilities[3] = {};

    calculate_softmax(
        prediction->logits,
        probabilities
    );

    int best_index = 0;

    for (int index = 1; index < 3; ++index) {
        if (
            probabilities[index]
            > probabilities[best_index]
        ) {
            best_index = index;
        }
    }

    prediction->class_index = best_index;
    prediction->class_name =
        CLASS_NAMES[best_index];

    prediction->confidence =
        probabilities[best_index];

    return ESP_OK;
}


void inference_deinit()
{
    delete model;

    model = nullptr;
    model_input = nullptr;
    model_output = nullptr;
}