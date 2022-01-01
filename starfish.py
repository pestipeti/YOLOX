from yolox.exp import Exp as MyExp
from yolox.utils.setup_env import increment_path


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- wandb config ---------------- #
        self.project_name = "TFStarfish"
        self.entity = "beluga_and_peter"

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 1
        self.width = 1

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 8
        self.input_size = (800, 1280)
        self.multiscale_range = 0
        # self.random_size = (20, 35)
        self.data_dir = None  # You have to add to the end of the train cli
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.degrees = 5.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 20
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 2
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1

        self.exp_name = "sf"
        self.exp_name = str(increment_path(f"YOLOX_outputs/{self.exp_name}", exist_ok=False)).split("/")[-1]

        # -----------------  testing config ------------------ #
        self.test_size = (800, 1280)
        self.test_conf = 0.1
        self.nmsthre = 0.6
