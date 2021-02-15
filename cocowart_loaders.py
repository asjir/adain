from pathlib import Path

from PIL import Image
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def ims_in(root, eval_frac=.2, seed=42):
    ims = list(Path(root).glob("**/*.jpg"))
    rs = ShuffleSplit(1, eval_frac, random_state=seed)
    train_index, eval_index = next(rs.split(ims))
    return ims[train_index], ims[eval_index]


def build_datasets(root_coco, root_wart, seed=42, eval_frac=.2,
                   batch_size=8):
    f = lambda x: ims_in(x, eval_frac=eval_frac, seed=seed)
    coco_train_ims, coco_eval_ims = f(root_coco)
    wart_train_ims, wart_eval_ims = f(root_wart)
    g = lambda x, y: DataLoader(CocoWArtDataset(x,y), batch_size=batch_size)
    return g(coco_train_ims, wart_train_ims), g(coco_eval_ims, wart_eval_ims)


class CocoWArtDataset(Dataset):
    def __init__(self, coco_ims, wart_ims, size=256, rescale_to=512) -> None:
        super().__init__()
        self.coco_ims = coco_ims
        self.wart_ims = wart_ims
        self.transforms = transforms.Compose(
            transforms.ToTensor,
            transforms.RandomCrop(size=size)
        )
        self.rescale_to = rescale_to
        self.i = 0

    def __len__(self):
        return len(self.coco_ims)
    
    def __getitem__(self, index):
        coco_im = self.coco_ims[index]
        wart_im = self.wart_ims[(self.i + index) % len(self.wart_ims)]
        f = lambda x: self.transforms(self.load_rescale(x))
        return f(coco_im), f(wart_im)
    
    def step(self):
        self.i += len(self.coco_ims)
        self.i %= len(self.wart_ims)
    
    def load_rescale(self, im_path):
        im = Image.open(im_path)
        H,W = im.size 
        scale = self.rescale_to / min(H,W)
        im.resize((int(H*scale), int(W*scale)), resample=Image.BICUBIC)
        return im