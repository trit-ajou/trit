import argparse

from .Pipeline import PipelineMgr
from .Utils import PipelineSetting, ImagePolicy
from .models.Utils import ModelMode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model1",
        type=str,
        default="skip",
        choices=["skip", "train", "inference"],
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="skip",
        choices=["skip", "train", "inference"],
    )
    parser.add_argument(
        "--model3",
        type=str,
        default="skip",
        choices=["skip", "train", "inference"],
    )
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=128)
    parser.add_argument("--use_noise", action="store_true")
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--max_objects", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--ckpt_interval", type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    setting = PipelineSetting()
    policy = ImagePolicy()

    args = parse_args()
    if args.model1 == "skip":
        setting.model1_mode = ModelMode.SKIP
    elif args.model1 == "train":
        setting.model1_mode = ModelMode.TRAIN
    elif args.model1 == "inference":
        setting.model1_mode = ModelMode.INFERENCE
    if args.model2 == "skip":
        setting.model2_mode = ModelMode.SKIP
    elif args.model2 == "train":
        setting.model2_mode = ModelMode.TRAIN
    elif args.model2 == "inference":
        setting.model2_mode = ModelMode.INFERENCE
    if args.model3 == "skip":
        setting.model3_mode = ModelMode.SKIP
    elif args.model3 == "train":
        setting.model3_mode = ModelMode.TRAIN
    elif args.model3 == "inference":
        setting.model3_mode = ModelMode.INFERENCE
    setting.use_amp = args.use_amp
    setting.num_workers = args.num_workers
    setting.num_images = args.num_images
    setting.use_noise = args.use_noise
    setting.margin = args.margin
    setting.max_objects = args.max_objects
    setting.epochs = args.epochs
    setting.batch_size = args.batch_size
    setting.lr = args.lr
    setting.weight_decay = args.weight_decay
    setting.vis_interval = args.vis_interval
    setting.ckpt_interval = args.ckpt_interval

    pipeline = PipelineMgr(setting, policy)
    pipeline.run()
