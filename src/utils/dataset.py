import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class StyleTransferDataset(Dataset):
    """
    Dataset para carregar imagens de conteúdo e estilo usando OpenCV.
    """
    def __init__(self, content_dir: str, style_dir: str, image_size: int = 256):
        """
        Args:
            content_dir (str): Caminho para a pasta com as fotos reais.
            style_dir (str): Caminho para a pasta com os frames de anime.
            image_size (int): O artigo LCCStyle especifica 300x300 para o treino.
        """
        self.content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png'))]
        self.style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir) if f.endswith(('.jpg', '.png'))]
        self.image_size = image_size

        # Garante que as listas tenham o mesmo tamanho repetindo a menor
        max_len = max(len(self.content_paths), len(self.style_paths))
        self.content_paths = (self.content_paths * (max_len // len(self.content_paths) + 1))[:max_len]
        self.style_paths = (self.style_paths * (max_len // len(self.style_paths) + 1))[:max_len]

    def _process_image(self, path: str) -> torch.Tensor:
        """Lê e processa a imagem com cv2."""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1) # [C, H, W]

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content_img = self._process_image(self.content_paths[idx])
        style_img = self._process_image(self.style_paths[idx])
        return content_img, style_img