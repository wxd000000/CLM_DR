import os.path as osp

from dl_lib.configs.segm_config import SemanticSegmentationConfig

_config_dict = dict(
    SEMI_MODE=True,
    MODEL=dict(
        WEIGHTS="",
        CAL_FLOPS=True,
        BACKBONE=dict(
            CELL_TYPE=['sep_conv_3x3', 'skip_connect'],
            LAYER_NUM=16,
            CELL_NUM_LIST=[2, 3, 4] + [4 for _ in range(13)],
            INIT_CHANNEL=64,
            MAX_STRIDE=32,
            SEPT_STEM=True,
            NORM="nnSyncBN",
            DROP_PROB=0.0,
        ),
        GATE=dict(
            GATE_ON=True,
            GATE_INIT_BIAS=1.5,
            SMALL_GATE=False,
        ),
        SEM_SEG_HEAD=dict(
            IN_FEATURES=['layer_0', 'layer_1', 'layer_2', 'layer_3'],
            NUM_CLASSES=21,
            IGNORE_VALUE=255,
            NORM="nnSyncBN",
            LOSS_WEIGHT=1.0,
        ),
        BUDGET=dict(
            CONSTRAIN=False,
            LOSS_WEIGHT=0.0,
            LOSS_MU=0.0,
            FLOPS_ALL=26300.0,
            UNUPDATE_RATE=0.4,
            WARM_UP=True,
        ),
    ),
    DATASETS=dict(
        TRAIN=("voc_2012_train", ),
        TEST=("voc_2012_val", ),
    ),
    SOLVER=dict(
        GPU_NUMS=1,
        LR_SCHEDULER=dict(
            NAME="PolyLR",
            POLY_POWER=0.9,
            MAX_ITER=100000,
        ),
        OPTIMIZER=dict(BASE_LR=0.1, ),
        IMS_PER_BATCH=4,
        CHECKPOINT_PERIOD=5000,
        GRAD_CLIP=5.0,
    ),
    INPUT=dict(
        MIN_SIZE_TRAIN=(320, 512, 768, 1024, 1280, 1536, 2048, ),
        MIN_SIZE_TRAIN_SAMPLING="choice",
        MAX_SIZE_TRAIN=4096,
        MIN_SIZE_TEST=1024,
        MAX_SIZE_TEST=2048,
        # FIX_SIZE_FOR_FLOPS=[768, 768],
        FIX_SIZE_FOR_FLOPS=[1024, 2048],
        CROP_PAD=dict(
            ENABLED=False,
            SIZE=[320, 320],
        ),
    ),
    TEST=dict(
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(320, 512, 768, 1024, 1280, 1536, 2048, ),
            MAX_SIZE=4096,
            FLIP=True,
        ),
        PRECISE_BN=dict(ENABLED=True),
    ),
    OUTPUT_DIR=osp.join('/mnt/change_code/DynamicRouting-master/playground/Dynamic/sup_voc_ly16_bs8_lr0.1/'),
)

class DynamicSemanticSegmentationConfig(SemanticSegmentationConfig):
    def __init__(self):
        super(DynamicSemanticSegmentationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = DynamicSemanticSegmentationConfig()
