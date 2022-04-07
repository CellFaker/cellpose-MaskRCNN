#Yu Yamaoka
#データセットのフォルダを指定してMaskRCNNが学習できるDatasetに変更する。

#Parameter Define
model_type = 'cyto'
chan = [0,0]#


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

    # save results as png
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

from PIL import Image

class ShapesDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        #dataset/blood/直下フォルダ名がクラス名に対応する．
        mask_class = ["1day","3day","5day","7day","other"]
        dataset_name = "cell_dataset"#保存されるモデル名に反映

        """データセットを登録"""
        i = 1
        for class_id in mask_class:
            self.add_class(dataset_name,i,class_id)
            i = i + 1

        """各クラスのフォルダ内画像を取得"""
        images = glob.glob(os.path.join(dataset_dir, "image", "*.png"))
        images = sorted(images)
        for class_id in mask_class:
            globals()["masks_" + class_id] =  glob.glob(os.path.join(dataset_dir, class_id, "*.png"))
            globals()["masks_" + class_id] = sorted(globals()["masks_" + class_id])

        """このfor文の書き換え出来ていません"""
        for image_path, mask_1day_path, mask_3day_path, mask_5day_path, mask_7day_path, mask_other_path in zip(images, masks_1day, masks_3day, masks_5day, masks_7day, masks_other):

            """各クラスのフォルダ内画像のPathを取得，名前チェックを行う"""
            image_path = pathlib.Path(image_path)
            temp_path = os.path.basename(os.path.normpath(image_path))#名前チェック用の確認用
            #print(image_path)#読み込んだ確認用
            for class_id in mask_class:
                globals()["mask_" + class_id + "_path"] = pathlib.Path(locals()["mask_" + class_id + "_path"])#画像ファイルの名前を取得
                assert temp_path == os.path.basename(os.path.normpath((locals()["mask_" + class_id + "_path"]))), locals()["mask_" + class_id + "_path"] + 'データセット名不一致です，名前のずれをチェック！'
                temp_path = os.path.basename(os.path.normpath((locals()["mask_" + class_id + "_path"])))

            """各クラスのフォルダ内画像サイズを取得，サイズチェックを行う"""
            image = Image.open(image_path)
            height = image.size[0]
            width = image.size[1]
            for class_id in mask_class:
                globals()["masks_" + class_id] = Image.open(locals()["mask_" + class_id + "_path"])
                temp_mask = Image.open(locals()["mask_" + class_id + "_path"])
                assert image.size == temp_mask.size, 'サイズ不一致！'

            """ここも書き換え出来ていません,mask_pathのところは手打ちで変更"""
            self.add_image(
                'cell_dataset',
                path=image_path,
                image_id=image_path.stem,
                mask_path=(mask_1day_path, mask_3day_path, mask_5day_path, mask_7day_path, mask_other_path),
                width=width, height=height)

    def load_mask(self, image_id):
        """マスクデータとクラスidを生成する"""
        #print(image_id)
        mask_class = ["1day","3day","5day","7day","other"]
        image_info = self.image_info[image_id]
        mask_1day_path, mask_3day_path, mask_5day_path, mask_7day_path, mask_other_path = image_info['mask_path']
        for i, class_id in enumerate(mask_class):
            tmp_mask = img_to_cellpose(str(locals()["mask_" + class_id + "_path"]))
            locals()["mask_" + class_id], locals()["cls_idxs_" + class_id] = obj_detection(tmp_mask, class_id=i+1)

        """複数のマスク配列を一つの配列に統合する．クラス配列も一つの配列に統合する．"""
        flag = 0
        for class_id in mask_class:
            if locals()["mask_"+class_id] is not None:
                if flag == 0:
                    mask = locals()["mask_" + class_id]
                    cls_idxs = locals()["cls_idxs_" + class_id]
                    flag = 1
                else:
                    mask = np.concatenate([mask, locals()["mask_" + class_id]], axis=2)
                    cls_idxs = np.concatenate([cls_idxs, locals()["cls_idxs_" + class_id]])
        if(flag==0):
            print('Error : ' + image_id + '全てのマスク画像が真っ黒です．rankエラーの原因になるかもしれません．')
        return mask, cls_idxs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'cell_dataset':
            return info
        else:
            super(self.__class__, self).image_reference(image_id)
