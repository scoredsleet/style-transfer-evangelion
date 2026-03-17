import torch
import torch.nn as nn
from .vgg_encoder import ContentEncoder, StyleEncoder
from .ltm import LTM
from .tfm import TFM
from .decoder import Decoder

class LCCStyleNet(nn.Module):
    """
    Classe principal que orquestra a arquitetura LCCStyle.
    
    Conecta os encoders, LTMs, TFMs e Decoder para realizar a transferência
    de estilo arbitrária de forma eficiente.
    """

    def __init__(self):
        super(LCCStyleNet, self).__init__()

        # Encoder do modelo
        self.content_encoder = ContentEncoder()
        self.style_encoder   = StyleEncoder()

        # Instâncias do LTM para processamento multi-nível
        # Canais baseados na arquitetura (32, 64, 128)
        self.ltm1 = LTM(in_channels=128, tfm_channels=128)
        self.ltm2 = LTM(in_channels=64, tfm_channels=64)
        self.ltm3 = LTM(in_channels=32, tfm_channels=32)
        
        # Decodificador final
        self.decoder = Decoder()
    
    def forward(self, content_img: torch.Tensor, style_img: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Realiza o fluxo completo de transferência de estilo (Forward Pass).

        Args:
            content_img (torch.Tensor): Imagem de conteúdo (B, 3, H, W).
            style_img (torch.Tensor): Imagem de estilo (B, 3, H, W).
            alpha (float): Trade-off entre conteúdo e estilo

        Returns:
            torch.Tensor: Imagem estilizada final.
        """

        # 1. Extração de características de estilo em 3 níveis
        s1, s2, s3 = self.style_encoder(style_img)
        cf = self.content_encoder(content_img)

        # 2. Geração Dinâmica de Pesos Convolucionais
        p1 = self.ltm1(s3)
        p2 = self.ltm2(s2)
        p3 = self.ltm3(s1)

        # 3. Mosaico de Reconstrução
        stylized_img = self.decoder(cf, p1, p2, p3)

        return stylized_img
