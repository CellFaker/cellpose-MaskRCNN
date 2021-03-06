#Yu Yamaoka
#img PATHを指定して推論，mask配列とclass配列を返す

#Path Define
model_path = "./logs/4+0723.h5"

from PIL import Image
import numpy as np

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.train_config import train_Config

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF



#dataset.load_mask()の代わり
def Inference(img_path):
    """
    Args:
        img_path:[string]
    """
    #For multi GPU 
    config_gpu = tf.ConfigProto()
    config_gpu.allow_soft_placement=True
    session = tf.Session(config=config_gpu)
    KTF.set_session(session)

    config = train_Config()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)
    model.load_weights(model_path, by_name=True)
    image = np.array(Image.open(img_path))
    results = model.detect([image], verbose=0)
    r = results[0]

    #trainで読み込めるように型変換
    mask = r['masks']
    mask = np.array(mask, dtype=np.uint8)
    mask[:, :, :] = np.where((mask[:, :, :]==False), 0 , 1) 

    cls_idxs = np.array(r['class_ids'] ,dtype=np.int32)

    return mask, cls_idxs

