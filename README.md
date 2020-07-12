# TFT-Shop

Counts champions in the shop for TFT Set 3. Use for 1920 x 1080 resolution only.

Code for generating data and training model: 
https://drive.google.com/drive/folders/1B5VwE-VHa-KsxYoR-GM3rqdwnygp9dq3?usp=sharing

Note: I won't be updating this since it's not feasible for me to recollect champion data every time a new set is released (data is hand-lebeled)

# Setup
Requires python 3.5-3.7:

pip install screeninfo

pip install opencv-python

pip install Pillow

pip install mss

pip install --upgrade tensorflow==1.15

pip install --upgrade keras

# How to Use
python image_detect.py

# Benchmarks
Intel Core i7, 2.8 GHz: 33 FPS

Intel Core i5, 1.4 GHz: 29 FPS

Note: FPS has been limited to 5 FPS in image_detect.py. Remove "sleep(0.2)" for maximum FPS, although it's not necessary since nobody rolls faster than 5 rolls/second.
