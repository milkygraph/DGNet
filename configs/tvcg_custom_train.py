model = dict(
    type="DGNet",
    encoder_channels=[64, 96,128, 160, 192, 224],
    decoder_channels=[224, 192, 160, 128, 96,64],
    radius = [0.1,0.2,0.4,0.6,0.8],
    dilations = [1,1,1,1,1],
    dropouts=[0.,0.,0.,0.,0.],
    max_sample = 30,
    temp_sample = 1000,
    use_pool = True,
    in_channels=16,
    num_classes=5 # 5 classes: left_arm, right_arm, legs, head, torso
)

dataroot = "datasets/custom_2_split100000"
batch_size = 2
feats = ["area","normal","center","color","angle","curvs"]
dataset = dict(
    train = dict(
        type = "Scannet",
        dataroot = dataroot,
        mode = "train",
        pattern = "*.obj",
        transforms = [
            dict(type="Distort"),
            dict(type="Rotation3"),
            dict(type="Normalize3")
        ],
        batch_size = batch_size,
        shuffle = True,
        color_aug = False,
        num_workers = 4,
        file_ext = ".obj",
        feats = feats,
    ),
    val = dict(
        type = "Scannet",
        dataroot = dataroot,
        pattern = "*.obj",
        mode = "val",
        transforms = [
            dict(type="Rotation3"),
            dict(type="Normalize3"),
        ],
        batch_size = 2,
        shuffle = False,
        color_aug = False,
        num_workers = 4,
        file_ext = ".obj",
        feats = feats
    )
)

optimizer = dict(
    type="Adam",
    lr = 1e-3,
    weight_decay = 1e-4
)

lr_scheduler = dict(
    type = "PolyLR",
    max_steps = 200,
)

logger = dict(
    type = "RunLogger"
)

checkpoint_interval = 5
log_interval = 1
eval_interval = 5
max_epoch = 200

iou_metric = True
ignore_index = 0
processor = "segmentation"

