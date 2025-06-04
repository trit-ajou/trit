from torch.utils.data import Dataset

from .TextedImage import TextedImage


class MangaDataset1(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len()

    def __getitem__(self, idx: int):
        pass


class MangaDataset2(Dataset):
    def __init__():
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]


class MangaDataset3(Dataset): # 초간단 model3 데이터셋
    def __init__(self, data: list[TextedImage]):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]