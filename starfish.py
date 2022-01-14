import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.utils.setup_env import increment_path
from yolox.utils.optimizer import Lookahead, RAdam


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- wandb config ---------------- #
        self.project_name = "TFStarfish"
        self.entity = "beluga_and_peter"

        # ---------------- model config ---------------- #
        self.num_classes = 1

        # self.depth = 0.67  # m
        # self.width = 0.75  # m

        self.depth = 1  # l
        self.width = 1  # l

        # self.depth = 1.33  # x
        # self.width = 1.25  # x

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 8
        self.input_size = (736, 1280)
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
        self.degrees = 0.0
        self.translate = 0.0
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 0.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 0
        self.max_epoch = 20
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.001 / 16.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 1
        self.min_lr_ratio = 0.1
        self.ema = True

        # None | low | high
        self.albu = "med"

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 50
        self.eval_interval = 1

        self.exp_name = "sf"
        self.exp_name = str(increment_path(f"YOLOX_outputs/{self.exp_name}", exist_ok=False)).split("/")[-1]

        # -----------------  testing config ------------------ #
        self.test_size = (736, 1280)
        self.test_conf = 0.1
        self.nmsthre = 0.4

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            # for k, v in self.model.named_modules():
            #     if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            #         pg2.append(v.bias)  # biases
            #     if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            #         pg0.append(v.weight)  # no decay
            #     elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            #         pg1.append(v.weight)  # apply decay

            # optimizer = torch.optim.SGD(
            #     pg0, lr=lr, momentum=self.momentum, nesterov=True
            # )
            # optimizer.add_param_group(
            #     {"params": pg1, "weight_decay": self.weight_decay}
            # )  # add pg1 with weight_decay
            # optimizer.add_param_group({"params": pg2})

            # self.optimizer = optimizer
            self.optimizer = Lookahead(
                RAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr),
                alpha=0.5,
                k=5
            )

        return self.optimizer
