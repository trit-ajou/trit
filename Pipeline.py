from torch import nn, optim
from torch.utils.data import DataLoader


from .datas.ImageLoader import ImageLoader
from .datas.Utils import TextedImage
from .datas.Dataset import MangaDataset1, MangaDataset2, MangaDataset3
from .models.Utils import ModelMode
from .Utils import PipelineSetting


class PipelineMgr:
    def __init__(self, setting: PipelineSetting):
        self.setting = setting

    def run(self, num_images: int):
        print("\n--- Starting Pipeline ---")

        # Step 1: Load Images
        print("[Pipeline] Loading Images")
        self.texted_images: list[TextedImage] = ImageLoader.load_images(
            path="images/clear/train",
            num_images=num_images,
            use_noise=self.setting.use_noise,
        )

        # Step 2: BBox Merging
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

        # Step 3: Model 1
        if self.setting.model1_mode != ModelMode.SKIP:
            dataset = MangaDataset1(self.texted_images, self.setting)
            dataloader = DataLoader(dataset)

            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")

            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")

            print("[Pipeline] Applying Model 1 output")

        else:
            print("[Pipeline] Skipping Model 1")

        # Step 4: Model 2

        # Step 5: Model 3

    @staticmethod
    def train_epoch(model, train_loader, valid_loader, optimizer, criterion, device):
        pass

    @staticmethod
    def eval(model, data_loader, criterion, device):
        pass
