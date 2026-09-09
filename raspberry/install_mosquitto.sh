#!/bin/bash

sudo apt update
sudo apt install -y mosquitto mosquitto-clients

echo "listener 1883
allow_anonymous true" | sudo tee /etc/mosquitto/conf.d/iot.conf > /dev/null

sudo systemctl enable mosquitto
sudo systemctl restart mosquitto

echo "Mosquitto installed and started."