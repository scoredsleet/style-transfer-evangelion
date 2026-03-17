import torch
import torch.nn as nn

from typing import Tuple

class ContentEncoder(nn.Module):
    """
    Content Encoder leve proposto no artigo LCCStyle.
    
    Responsável por mapear a imagem de conteúdo para o domínio de características,
    focando na retenção de forma (shape retention).
    """

    def __init__(self):
        """
        Inicializa o Content Encoder com 3 camadas convolucionais seguidas de
        Instance Normalization e ReLU, conforme a seção III-B do artigo.
        """
        super(ContentEncoder, self).__init__()

        # Camada 1: Conv2d_3_32_1 (Kernel 3, Out 32, Stride 1) 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(32)

        # Camada 2: Conv2d_3_64_2 (Kernel 3, Out 64, Stride 2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64)

        # Camada 3: Conv2d_3_128_2 (Kernel 3, Out 128, Stride 2) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai as características de conteúdo da imagem de entrada.

        Args:
            x (torch.Tensor): Tensor 4D da imagem de conteúdo no formato (B, 3, H, W).

        Returns:
            torch.Tensor: Mapa de características profundo no formato (B, 128, H/4, W/4).
        """
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        return x

class StyleEncoder(nn.Module):
    """
    Style Encoder independente proposto no artigo LCCStyle.
    
    Utiliza processamento multi-nível para extrair características de estilo
    que alimentarão diferentes módulos LTM.
    """

    def __init__(self):
        """
        Inicializa o Style Encoder com três conjuntos de camadas convolucionais
        conforme especificado na seção III-B do artigo.
        """
        super(StyleEncoder, self).__init__()

        # Conjunto 1: 3x Conv2d_3_32_1 
        self.set1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(True)
        )

        # Conjunto 2: 1x Conv2d_3_64_2 e 2x Conv2d_3_64_1
        self.set2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.ReLU(True)
        )
        
        # Conjunto 3: 1x Conv2d_3_128_2 e 2x Conv2d_3_128_1 
        self.set3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extrai características de estilo em três níveis diferentes (Multi-Level).

        Args:
            x (torch.Tensor): Tensor 4D da imagem de estilo no formato (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tripla contendo:
                - feat1: Estilo de baixo nível (B, 32, H, W).
                - feat2: Estilo de nível médio (B, 64, H/2, W/2).
                - feat3: Estilo de alto nível (B, 128, H/4, W/4).
        """
        feat1 = self.set1(x)
        feat2 = self.set2(feat1)
        feat3 = self.set3(feat2)
        return feat1, feat2, feat3