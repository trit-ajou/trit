import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader


from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.Dataset import MangaDataset1, MangaDataset2, MangaDataset3
from .models.Model1 import Model1
from .models.Model2 import Model2
from .models.Model3 import Model3
from .models.Utils import ModelMode
from .Utils import PipelineSetting


class PipelineMgr:
    def __init__(self, setting: PipelineSetting):
        self.model = None
        self.setting = setting

    def run(self):
        print("[Pipeline] Starting Pipeline")

        # Step 1: Load Images
        print("[Pipeline] Loading Images")
        self.texted_images: list[TextedImage] = ImageLoader.load_images(self.setting)

        # Step 2: BBox Merging
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

        # Step 3: Model 1
        if self.setting.model1_mode != ModelMode.SKIP:
            self.model = Model1()
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                # TODO: model 1 train, viz

            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
                # TODO: model 1 inference, viz, apply

        else:
            print("[Pipeline] Skipping Model 1")

        # Step 4: Model 2
        if self.setting.model2_mode != ModelMode.SKIP:
            self.model = Model2()
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                # TODO: model 1 train, viz

            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
                # TODO: model 1 inference, viz, apply

        else:
            print("[Pipeline] Skipping Model 1")

        # Step 5: Model 3
        if self.setting.model3_mode != ModelMode.SKIP:
            self.model = Model3()
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                # TODO: model 1 train, viz

            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
                # TODO: model 1 inference, viz, apply

        else:
            print("[Pipeline] Skipping Model 1")

    def train(
        self,
        data_loader,
        criterion,
        optimizer,
        scaler,
    ):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        preds = []

        for inputs, targets in tqdm(data_loader, leave=False):
            inputs = inputs.to(self.setting.device)
            targets = targets.to(self.setting.device)
            optimizer.zero_grad()

            with torch.autocast(
                device_type=self.setting.device,
                enabled=self.setting.use_amp,
            ):
                pred = self.model(inputs)
                loss = criterion(pred, targets)
                preds.append(pred)

            train_loss += loss.item()
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        avg_loss = train_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def eval(self, data_loader, criterion):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        preds = []
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, leave=False):
                inputs = inputs.to(self.setting.device)
                targets = targets.to(self.setting.device)

                with torch.autocast(
                    device_type=self.setting.device,
                    enabled=self.setting.use_amp,
                ):
                    pred = self.model(inputs)
                    loss = criterion(pred, targets)
                    preds.append(pred)

                test_loss += loss.item()
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        avg_loss = test_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy, pred

    def get_optim_AdamW(self, params):
        return torch.optim.AdamW(
            params, lr=self.setting.lr, weight_decay=self.setting.weight_decay
        )

    def get_scaler(self):
        return torch.amp.GradScaler(self.setting.device, enabled=self.setting.use_amp)

    def get_train_valid_loader(self, split_ratio=0.2):
        if self.model is None:
            return None

        split_idx = int(len(self.texted_images) * split_ratio)
        if isinstance(self.model, Model1):
            train_set = MangaDataset1(self.texted_images[split_idx:])
            valid_set = MangaDataset1(self.texted_images[:split_idx])
        elif isinstance(self.model, Model2):
            train_set = MangaDataset2(self.texted_images[split_idx:])
            valid_set = MangaDataset2(self.texted_images[:split_idx])
        elif isinstance(self.model, Model3):
            train_set = MangaDataset3(self.texted_images[split_idx:])
            valid_set = MangaDataset3(self.texted_images[:split_idx])

        train_loader = DataLoader(
            train_set,
            batch_size=self.setting.batch_size,
            num_workers=self.setting.num_workers,
            pin_memory=True,
            persistent_workers=self.setting.num_workers > 0,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=self.setting.batch_size,
            num_workers=self.setting.num_workers,
            pin_memory=True,
            persistent_workers=self.setting.num_workers > 0,
        )
        return train_loader, valid_loader

    def get_test_loader(self):
        if self.model is None:
            return None

        if isinstance(self.model, Model1):
            test_set = MangaDataset1(self.texted_images)
        elif isinstance(self.model, Model2):
            test_set = MangaDataset2(self.texted_images)
        elif isinstance(self.model, Model3):
            test_set = MangaDataset3(self.texted_images)

        return DataLoader(
            test_set,
            batch_size=self.setting.batch_size,
            num_workers=self.setting.num_workers,
            pin_memory=True,
            persistent_workers=self.setting.num_workers > 0,
        )
