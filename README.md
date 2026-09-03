# Smart Visual Alarm System

IoT and TinyML project for visual monitoring of a mountain house using an ESP32-S3-EYE and a Raspberry Pi.

The system performs image classification directly on the ESP32 and distinguishes three classes:

- `animal`
- `empty`
- `person`

Images are processed locally. When a person or an animal is detected, the ESP32 sends the captured JPEG image to the Raspberry Pi through HTTP and publishes the prediction through MQTT. Node-RED receives the event and sends a Telegram notification with the captured image.

## System Architecture

```text
ESP32-S3-EYE
      |
      | Camera frame 320 x 240
      v
Image preprocessing
      |
      | Resize to 48 x 48
      v
MCUNet INT8 inference
      |
      +-----------------------------+
      |                             |
      | MQTT prediction             | HTTP JPEG upload
      v                             v
Mosquitto                     Flask image receiver
      |                             |
      |                       latest.jpg
      v                             |
Node-RED <--------------------------+
      |
      +--> Event logging
      +--> Dashboard
      +--> Telegram notification
```

All image classification is performed on-device. Images are not sent to a cloud service for inference.

## Hardware

- ESP32-S3-EYE
- OV2640 camera
- Raspberry Pi 400
- Wi-Fi network

The ESP32-S3-EYE has 8 MB PSRAM and runs the final quantized neural network using ESP-DL.

## Software

### ESP32

- ESP-IDF 6.0.2
- ESP-DL
- esp32-camera
- ESP-MQTT
- ESP HTTP Client

### Raspberry Pi

- Raspberry Pi OS
- Mosquitto MQTT broker
- Node-RED
- Flask
- Telegram Bot

### Machine Learning

- Python
- PyTorch
- torchvision
- scikit-learn
- matplotlib

## Dataset

The final dataset contains 400 images divided into three classes.

```text
dataset/
├── calibration/
├── train/
│   ├── animal/
│   ├── empty/
│   └── person/
├── val/
│   ├── animal/
│   ├── empty/
│   └── person/
└── test/
    ├── animal/
    ├── empty/
    └── person/
```

Final split:

| Split | Images |
|---|---:|
| Train | 280 |
| Validation | 60 |
| Test | 60 |
| **Total** | **400** |

The `calibration` set is used for INT8 quantization of MCUNet.

Dataset images are kept locally and are not included in the public repository.

The `training` directory also contains the scripts used during the original dataset preparation, including image validation, manual labeling, dataset splitting and augmentation.

## Image Preprocessing

The neural networks use RGB images resized to:

```text
48 x 48
```

The images are normalized using ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training data augmentation includes small rotations, translations, scaling, horizontal flipping, brightness and contrast changes, and occasional Gaussian blur.

## Models

Three models were trained and compared using the same dataset.

### MLP

A multilayer perceptron was used as the baseline model.

### MCUNet-style model

A compact convolutional neural network based on MCUNet design principles and inverted residual blocks.

The architecture was manually defined for this project and was not generated using the official TinyNAS architecture search.

### TinyViT-style model

A compact Vision Transformer designed for comparison with the convolutional MCUNet model.

It is a custom TinyViT-style architecture and not the official Microsoft TinyViT implementation.

## Model Comparison

| Model | Test Accuracy | Macro F1 | Parameters | Checkpoint Size | Inference Time* |
|---|---:|---:|---:|---:|---:|
| MLP | 81.67% | 0.8257 | 889,411 | 3.40 MB | 4.495 ms |
| MCUNet | **90.00%** | **0.9027** | 58,563 | 0.28 MB | 5.845 ms |
| TinyViT | 73.33% | 0.7398 | **46,563** | **0.19 MB** | 5.530 ms |

\*Inference times in this table were measured during model evaluation on the development machine and are not ESP32 inference benchmarks.

MCUNet was selected for deployment because it achieved the highest classification accuracy and Macro F1 while keeping the model compact enough for embedded deployment.

## INT8 Quantization

The selected MCUNet model is quantized to INT8 using a representative calibration dataset.

```text
PyTorch MCUNet
      |
      v
INT8 quantization
      |
      v
mcunet_int8.espdl
      |
      v
ESP32-S3-EYE
```

The deployment graph removes operations that caused compatibility problems during the first export attempts and produces an ESP-DL model directly from the PyTorch network.

The final model is stored in:

```text
firmware/esp32_s3_eye_alarm/main/model/mcunet_int8.espdl
```

## ESP32 Firmware

The firmware performs the following operations continuously:

1. Capture a JPEG frame from the OV2640 camera at 320 x 240.
2. Decode the JPEG image.
3. Resize the image to 48 x 48.
4. Normalize and quantize the input.
5. Run MCUNet INT8 inference using ESP-DL.
6. Compute the predicted class and confidence.
7. Publish the prediction through MQTT.
8. For `person` or `animal`, upload the JPEG image to the Raspberry Pi.

The classes used by the deployed model are:

```text
0 = animal
1 = empty
2 = person
```

## MQTT

The ESP32 publishes predictions to:

```text
smartalarm/prediction
```

Example:

```json
{
  "class": "person",
  "confidence": 0.9232
}
```

Mosquitto runs on the Raspberry Pi on the standard MQTT port:

```text
1883
```

## HTTP Image Upload

When an alarm class is detected, the ESP32 sends the original JPEG frame to:

```text
POST /upload-alert
Content-Type: image/jpeg
```

The Flask receiver stores the most recent image as:

```text
runtime/images/latest.jpg
```

The endpoint:

```text
GET /health
```

can be used to verify that the receiver is running.

Runtime images and logs are excluded from Git.

## Node-RED

Node-RED subscribes to:

```text
smartalarm/prediction
```

The flow:

- receives MQTT predictions;
- adds timestamps;
- records events;
- updates the dashboard;
- processes the three predicted classes;
- sends Telegram alerts for `person` and `animal`;
- attaches the latest image received from the ESP32.

The Node-RED flow is stored in:

```text
node-red/flows.json
```

## Telegram Alerts

For a relevant detection, the user receives a Telegram notification containing:

- detected class;
- confidence;
- timestamp;
- captured image.

No alert is generated for an `empty` scene.

Telegram credentials and other private configuration values are not stored in the repository.

## Repository Structure

```text
Smart-Visual-Alarm-System/
├── dataset/
├── esp32/
├── firmware/
│   └── esp32_s3_eye_alarm/
├── node-red/
│   └── flows.json
├── raspberry/
│   ├── image_receiver.py
│   └── install_mosquitto.sh
├── results/
│   ├── comparison/
│   ├── mcunet/
│   ├── mlp/
│   └── tinyvit/
├── runtime/
├── training/
├── README.md
├── requirements.txt
└── .gitignore
```

## Main Training Scripts

```text
training/
├── augmentation.py
├── dataset.py
├── train_mlp.py
├── evaluate_mlp.py
├── train_mcunet.py
├── evaluate_mcunet.py
├── quantize_mcunet.py
├── train_tiny_vit.py
├── evaluate_tiny_vit.py
└── compare_models.py
```

Additional scripts document the dataset collection, validation, labeling and preparation process.

## Raspberry Pi Setup

Install and start Mosquitto:

```bash
bash raspberry/install_mosquitto.sh
```

Start the HTTP image receiver:

```bash
python3 raspberry/image_receiver.py
```

Check the receiver:

```bash
curl http://localhost:5000/health
```

Node-RED must be running with the flow contained in `node-red/flows.json`.

## ESP32 Build and Flash

Activate ESP-IDF:

```bash
source ~/esp/esp-idf/export.sh
```

Enter the firmware directory:

```bash
cd firmware/esp32_s3_eye_alarm
```

Build:

```bash
idf.py build
```

Flash and open the serial monitor:

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

The serial port may be different depending on the system.

Wi-Fi credentials, MQTT broker address and HTTP receiver address are configured locally in:

```text
firmware/esp32_s3_eye_alarm/main/secrets.hpp
```

This file is excluded from Git. An example configuration is provided in:

```text
secrets.example.hpp
```

## Final Integration Test

The complete system has been tested successfully with the ESP32-S3-EYE and Raspberry Pi.

Verified operations:

- camera initialization;
- JPEG image acquisition at 320 x 240;
- MCUNet INT8 model loading;
- on-device inference;
- `animal`, `empty` and `person` classification;
- Wi-Fi connection;
- MQTT publication;
- HTTP JPEG upload;
- image storage on the Raspberry Pi;
- Node-RED event processing;
- Telegram notification with the captured image.

The final tested pipeline is:

```text
ESP32-S3-EYE
      ↓
MCUNet INT8 inference
      ↓
person / animal detected
      ↓
HTTP image upload + MQTT event
      ↓
Raspberry Pi
      ↓
Flask + Mosquitto
      ↓
Node-RED
      ↓
Telegram alert with image
```

## Project Status

- [x] ESP32-S3-EYE camera configured
- [x] Dataset collected and labeled
- [x] Dataset cleaned and split
- [x] Data augmentation implemented
- [x] MLP trained and evaluated
- [x] TinyViT-style model trained and evaluated
- [x] MCUNet-style model trained and evaluated
- [x] Models compared
- [x] MCUNet selected for deployment
- [x] MCUNet quantized to INT8
- [x] ESP-DL model deployed on ESP32-S3-EYE
- [x] On-device inference tested
- [x] MQTT communication tested
- [x] HTTP image upload tested
- [x] Node-RED dashboard and event processing tested
- [x] Telegram notification with image tested
- [x] Complete ESP32 → Raspberry Pi → Telegram pipeline tested
- [ ] Final test in the mountain house environment

## Notes

This project was developed for the Internet of Things course at the University of Trento.

The objective is to combine TinyML inference on a resource-constrained embedded device with standard IoT communication protocols and an edge gateway, while keeping image classification local to the ESP32.