Currently only configured for 1920x1080 resolution monitors.

Code: https://drive.google.com/drive/folders/1B5VwE-VHa-KsxYoR-GM3rqdwnygp9dq3?usp=sharing

Runs well on CPU:
Intel Core i7, 2.8 GHz: 33 FPS
Intel Core i5, 1.4 GHz: 29 FPS

Requires python 3.5-3.7:
pip install screeninfo
pip install opencv-python
pip install Pillow
pip install mss
pip install --upgrade tensorflow==1.15
pip install --upgrade keras

run:
python image_detect.py