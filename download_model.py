import os
import urllib.request
import tarfile

MODEL_URL = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz'
MODEL_DIR = 'ssd_mobilenet_v2_coco_2018_03_29'
TAR_NAME = 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz'

if not os.path.exists(MODEL_DIR):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, TAR_NAME)

    print("Extracting model...")
    with tarfile.open(TAR_NAME) as tar:
        tar.extractall()
    os.remove(TAR_NAME)
    print("Model ready!")
else:
    print("Model already exists.")
