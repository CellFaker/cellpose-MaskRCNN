#Yu Yamaoka
#データセットのフォルダを指定してMaskRCNNが学習できるDatasetに変更する。

#Parameter Define
model_type = 'cyto'
chan = [0,0]#

#Path Define
mask_class = ["0day","3day"]
dataset_name = "cell_dataset"#保存されるモデル名に反映

#Function Define
from cellpose import models, io
import matplotlib.pyplot as plt
import numpy as np

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
def img_to_cellpose(img_path):
    """
    Input:
        img_path : (string) Image file PATH
    Return:
        saved inference data at file PATH.
    """
    model = models.Cellpose(gpu=False, model_type=model_type)
    img = io.imread(img_path)
    mask, flows, styles, diams = model.eval(img, diameter=None, channels=chan)

    # save results so you can load in gui
    #io.masks_flows_to_seg(img, masks, flows, diams, img_path, chan)

    #save results as png
    #plt.imsave("test.png",masks)

    return mask

#mask画像をMaskRCNNが読み込めるデータセットにする。
def obj_detection(mask, class_id:int):
    """
    Input:
        mask : [width, height](ndarray), image data
        class_id : int , class id(ex : 1day -> 1)
    Return:
        mask : [width, height, n], n is object num.
        cls_idxs : [n(int)]
    """
    data = mask
    labels = []
    for label in np.unique(data):
        #: ラベルID==0は背景
        if label == 0:
            continue
        else:
            labels.append(label)

    if len(labels) == 0:
        #: 対象オブジェクトがない場合はNone
        return None, None
    else:
        mask = np.zeros((mask.shape)+(len(labels),), dtype=np.uint8)
        for n, label in enumerate(labels):
            mask[:, :, n] = np.uint8(data == label)
        cls_idxs = np.ones([mask.shape[-1]], dtype=np.int32) * class_id

        return mask, cls_idxs


from mrcnn import utils
from mrcnn.model import log
import os
import glob
import pathlib
import itertools

from PIL import Image

class ShapesDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        #dataset/blood/直下フォルダ名がクラス名に対応する．

        """データセットでのクラスを1から登録"""
        for i, class_id in enumerate(mask_class):
            self.add_class(dataset_name, i+1, class_id) 

        """各クラスのフォルダ内画像Pathを取得"""
        #image_files = [""]*len(mask_class)
        for i, class_name in enumerate(mask_class):
            image_paths = glob.glob(os.path.join(dataset_dir, class_name, "*.png"))

            for image_path in image_paths:
                image_path = pathlib.Path(image_path)
                image = Image.open(image_path)
                height = image.size[0]
                width = image.size[1]

                self.add_image(
                    dataset_name,
                    path=image_path,
                    image_id=image_path.stem,
                    mask_path=(image_path),
                    width=width, height=height)

    def load_mask(self, image_id):
        """マスクデータとクラスidを生成する"""
        #print(image_id)
        image_info = self.image_info[image_id]
        mask_path = image_info['mask_path']
        for i, class_name in enumerate(mask_class):
            mask_path = str(mask_path)
            if class_name in mask_path:
                class_id = i + 1
        mask = img_to_cellpose(mask_path)
        mask, cls_idxs = obj_detection(mask, class_id=class_id)
        #print(mask_path)
        #print(cls_idxs)

        return mask, cls_idxs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'cell_dataset':
            return info
        else:
            super(self.__class__, self).image_reference(image_id)
