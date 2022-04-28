'''モジュールの説明
MaskRCNN学習時のハイパーパラメータを設定する．

Args:
    LEARNING_RATE :
    validation errorがより少なくなるように設定する。
    validation errorの減少するスピードが遅ければ(①)learning rateを増やし、
    validation errorが増加してしまっているなら(②)learning rateを減らすなど

    GPU_COUNT:
    学習時に使用するグラフィックボードの数。

    IMAGES_PER_GPU:
    一つのGPUが担当する画像の枚数．12GBのGPUだと1024×1024の画像を2枚扱える．

'''
from mrcnn.config import Config

class train_Config(Config):

    #: config名
    NAME = "cell_dataset"

    LEARNING_RATE = 0.001

    GPU_COUNT = 2

    #: バッチあたり画像数 (GPUのメモリが大きいなら増やしてもよい
    IMAGES_PER_GPU = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 896
    IMAGE_MAX_DIM = 896

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)  # anchor side in pixels

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 800

    # クラス数　= 背景 + 検出クラス数
    NUM_CLASSES = 1 + 2

    # エポックあたりステップ数
    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = 5

    # 提案領域のconfidenceが90%以下なら物体検出フェイズをスキップ
    #DETECTION_MIN_CONFIDENCE = 0.9
