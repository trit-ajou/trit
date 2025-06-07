import argparse
import os

from .Pipeline import PipelineMgr
from .Utils import PipelineSetting, ImagePolicy, TimgGeneration
from .models.Utils import ModelMode
from dataclasses import dataclass, field
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model1",
        type=str,
        default="default",
        choices=["skip", "train", "inference"],
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="default",
        choices=["skip", "train", "inference"],
    )
    parser.add_argument(
        "--model3",
        type=str,
        default="skip",
        choices=["skip", "train", "inference", "pretrained", "pretrained-train"],
    )
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=PipelineSetting.num_workers)
    parser.add_argument("--num_images", type=int, default=PipelineSetting.num_images)
    parser.add_argument("--use_noise", action="store_true")

    parser.add_argument("--margin", type=int, default=PipelineSetting.margin)
    parser.add_argument("--max_objects", type=int, default=PipelineSetting.max_objects)
    parser.add_argument("--epochs", type=int, default=PipelineSetting.epochs)
    parser.add_argument("--batch_size", type=int, default=PipelineSetting.batch_size)
    parser.add_argument("--lr", type=float, default=PipelineSetting.lr)
    parser.add_argument("--weight_decay", type=float, default=PipelineSetting.weight_decay)
    parser.add_argument("--vis_interval", type=int, default=PipelineSetting.vis_interval)
    parser.add_argument("--ckpt_interval", type=int, default=PipelineSetting.ckpt_interval)
    parser.add_argument("--timg_generation", type=str, default="generate_only", choices=["generate_only", "generate_save", "use_saved", "test"])
    # Model 3 training Options
    parser.add_argument("--images_dir", type=str, default="trit/datas/images/clear")  
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_weight_path", type = str, default="trit/models/lora")
    parser.add_argument("--mask_weight", type=float, default=2.0)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    setting = PipelineSetting()
    policy = ImagePolicy(text_color_is_random=False,
                         fixed_text_color_options=[(0, 0, 0),(0, 0, 0)],
                         opacity_range=(255,255),
                         stroke_color_is_random=False,
                         stroke_prob=1,
                         fixed_stroke_color_options=[(255,255,255),(255,255,255)],
                         shadow_prob=0,

                         )
    # print(policy)
    args = parse_args()
    if args.timg_generation == "generate_only":
        setting.timg_generation = TimgGeneration.generate_only
    elif args.timg_generation == "generate_save":
        setting.timg_generation = TimgGeneration.generate_save
    elif args.timg_generation == "use_saved":
        setting.timg_generation = TimgGeneration.use_saved
    elif args.timg_generation == "test":
        setting.timg_generation = TimgGeneration.test
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
    elif args.model3 == "pretrained":
        setting.model3_mode = ModelMode.PRETRAINED
    elif args.model3 == "pretrained-train":
        setting.model3_mode = ModelMode.PRETRAINED_TRAIN

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
    setting.clear_img_dir = args.images_dir
    setting.lora_rank = args.lora_rank
    setting.lora_alpha = setting.lora_rank * 2
    setting.lora_weight_path = args.lora_weight_path
    
    
    

    pipeline = PipelineMgr(setting, policy)
    pipeline.run()
