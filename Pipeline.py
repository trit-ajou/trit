import tqdm
import torch
from torch.utils.data import DataLoader


from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.Dataset import MangaDataset
from .models.Model1 import Model1
from .models.Model2 import Model2
from .models.Model3 import Model3
from .models.Utils import ModelMode
from .Utils import PipelineSetting, ImagePolicy


class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.model = None
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)

    def run(self):
        print("[Pipeline] Starting Pipeline")

        # Step 1: Load Images
        print("[Pipeline] Loading Images")
        self.texted_images: list[TextedImage] = self.imageloader.load_images(
            self.setting.num_images
        )
        self.texted_images[0].visualize(self.setting.output_img_dir, "step1.png")

        # Step 2: BBox Merging
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)
        self.texted_images[0].visualize(self.setting.output_img_dir, "step2.png")

        # Step 3: Model 1
        if self.setting.model1_mode != ModelMode.SKIP:
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                # TODO: model 1 train, viz
            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
                # TODO: model 1 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 1")

        # Step 4: Model 1 output apply
        if self.setting.model1_mode == ModelMode.INFERENCE:
            pass

        # Step 5: Model 2
        if self.setting.model2_mode != ModelMode.SKIP:
            # Prepare datas
            texted_images_splitted: list[TextedImage] = []
            for texted_image in self.texted_images:
                texted_images_splitted.extend(
                    texted_image.split_margin_crop(self.setting.margin)
                )
            texted_images_splitted[0].visualize(
                self.setting.output_img_dir, "step5.png"
            )
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 2")
                # TODO: model 2 train, viz
            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 2 Inference")
                # TODO: model 2 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 2")

        # Step 6: Model 2 output apply
        if self.setting.model2_mode == ModelMode.INFERENCE:
            pass

        # Step 7: Model 3
        if self.setting.model3_mode != ModelMode.SKIP:
            # Prepare datas
            texted_images_splitted: list[TextedImage] = []
            for texted_image in self.texted_images:
                texted_images_splitted.extend(
                    texted_image.split_center_crop(self.setting.model3_input_size)
                )
            texted_images_splitted[0].visualize(
                self.setting.output_img_dir, "step7.png"
            )
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 3")
                # TODO: model 3 train, viz
            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                # TODO: model 3 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 3")

        # Step 8: Model 3 output apply
        if self.setting.model3_mode == ModelMode.INFERENCE:
            pass

        # Step 9: Final viz
        self.texted_images[0].visualize(self.setting.output_img_dir, "step9.png")
