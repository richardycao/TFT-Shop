import cv2
from keras.optimizers import Adamax
from keras.models import model_from_json
import numpy as np
import os
from PIL import Image
from mss import mss
from time import sleep
from screeninfo import get_monitors

##################################################################################

json_name = 'cnn_baseline_EPOCHS50.json'
h5_name = 'cnn_baseline_EPOCHS50.h5'

# load json and create model
json_file = open('model\\' + json_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('model\\' + h5_name)
print("Loaded model from disk")
# compile model
opt = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

###################################################################################

class_list = []
with open('classes.txt', 'r') as f:
    class_list = f.read().split('\n')[:-1]
class_dict = {v: k for v, k in enumerate(class_list)}

# Dell XPS 15 - 1920x1080
mx, my, mw, mh = 0, 0, 0, 0
for m in get_monitors():
    mx, my, mw, mh = m.x, m.y, m.width, m.height
    break

bounding_box1 = {'top': 930, 'left': 530, 'width': 128, 'height': 128}
bounding_box2 = {'top': 930, 'left': 731, 'width': 128, 'height': 128}
bounding_box3 = {'top': 930, 'left': 932, 'width': 128, 'height': 128}
bounding_box4 = {'top': 930, 'left': 1133, 'width': 128, 'height': 128}
bounding_box5 = {'top': 930, 'left': 1334, 'width': 128, 'height': 128}
sct1 = mss()
sct2 = mss()
sct3 = mss()
sct4 = mss()
sct5 = mss()

prev_classes = [0,0,0,0,0]

while True: #for _ in range(100):
    im1 = np.array(sct1.grab(bounding_box1))
    im2 = np.array(sct2.grab(bounding_box2))
    im3 = np.array(sct3.grab(bounding_box3))
    im4 = np.array(sct4.grab(bounding_box4))
    im5 = np.array(sct5.grab(bounding_box5))

    im1 = np.flip(im1[:, :, :3], 2)
    im2 = np.flip(im2[:, :, :3], 2)
    im3 = np.flip(im3[:, :, :3], 2)
    im4 = np.flip(im4[:, :, :3], 2)
    im5 = np.flip(im5[:, :, :3], 2)

    ims = np.array([im1, im2, im3, im4, im5])
    result = model.predict_classes(np.reshape(ims, (ims.shape[0],128,128,3)))

    classes = []
    for r in range(len(result)):
        classes.append(class_dict[result[r]])

    sims = np.sum([classes[i]==prev_classes[i] for i in range(len(classes))])
    if (sims <= 3) and ('0' not in classes):
        print(classes)
        prev_classes = [c for c in classes]
    sleep(0.2)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break


