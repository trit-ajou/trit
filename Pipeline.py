import torch
from tqdm import tqdm
from copy import copy
from torch.utils.data import DataLoader, random_split


from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.Dataset import MangaDataset1
from .models.Utils import ModelMode
from .models.Model1 import Model1
from .Utils import PipelineSetting, ImagePolicy


class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)

    def run(self):
        ################################################### Step 1: Load Images ##############################################
        print("[Pipeline] Loading Images")
        self.texted_images: list[TextedImage] = self.imageloader.load_images(
            self.setting.num_images, self.setting.clear_img_dir
        )

        ################################################### Step 2: BBox Merge ###############################################
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

        ################################################### Step 3: Model 1 ##################################################
        if self.setting.model1_mode != ModelMode.SKIP:
            texted_images_for_model1 = [
                copy(texted_image) for texted_image in self.texted_images
            ]
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                # TODO: model 1 train, viz

            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
                # TODO: model 1 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 1")

        ################################################### Step 4: Model 1 output apply #####################################
        if self.setting.model1_mode == ModelMode.INFERENCE:
            pass

        ################################################### Step 5: Model 2 ##################################################
        if self.setting.model2_mode != ModelMode.SKIP:
            texted_images_for_model2 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_margin_crop(self.setting.margin)
            ]
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 2")
                # TODO: model 2 train, viz
            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 2 Inference")
                # TODO: model 2 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 2")

        ################################################### Step 6: Model 2 output apply #####################################
        if self.setting.model2_mode == ModelMode.INFERENCE:
            pass

        ################################################### Step 7: Model 3 ##################################################
        if self.setting.model3_mode != ModelMode.SKIP:
            texted_images_for_model3 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_center_crop(
                    self.setting.model3_input_size
                )
            ]
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 3")
                # TODO: model 3 train, viz
            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                # TODO: model 3 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 3")

        ################################################### Step 8: Model 3 output apply #####################################
        if self.setting.model3_mode == ModelMode.INFERENCE:
            pass
