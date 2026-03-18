import torch
import torch.nn as nn
import torch.nn.functional as F
from .tfm import TFM

class Decoder(nn.Module):
    """
    Decoder leve proposto no artigo LCCStyle.
    
    Responsável por mapear o mapa de características estilizado de volta para o
    domínio da imagem (pixels RGB). É simétrico ao Content Encoder.
    """

    def __init__(self):
        """
        Inicializa o Decoder com camadas de convolução e upsampling.
        As camadas seguem a nomenclatura D1, D2, D3 e UL (Upsample Layer) do artigo.
        """
        super(Decoder, self).__init__()
        self.tfm = TFM()

        # De acordo com a seção III-C e III-D do artigo:
        
        # Camada D1: Processa o nível mais profundo (recebe do TFM1)
        # Conv2d_3_64_1 (Kernel 3, Out 64, Stride 1)
        self.d1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        
        # Camada D2: Processa o nível intermediário (recebe do TFM2 + UL1)
        # Conv2d_3_32_1 (Kernel 3, Out 32, Stride 1)
        self.d2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        
        # Camada D3: Camada final de reconstrução RGB
        # Conv2d_3_3_1 (Kernel 3, Out 3, Stride 1)
        self.d3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.relu = nn.ReLU(inplace=True)

    def upsample_x2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza o aumento de escala de 2x usando interpolação por vizinho mais próximo.
        
        Args:
            x (torch.Tensor): Tensor de entrada.
            
        Returns:
            torch.Tensor: Tensor com o dobro da altura e largura.
        """
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, cf: torch.Tensor, p1: tuple, p2: tuple, p3: tuple) -> torch.Tensor:
        """
        Reconstrói a imagem estilizada usando processamento multi-nível.
        Os TFMs (TFM1, TFM2, TFM3) são inseridos antes das camadas do decoder.

        Args:
            p1 (torch.Tensor): Saída do TFM1 (Nível profundo - 128 canais).
            p2 (torch.Tensor): Saída do TFM2 (Nível médio - 64 canais).
            p3 (torch.Tensor): Saída do TFM3 (Nível inicial - 32 canais).

        Returns:
            torch.Tensor: Imagem final estilizada (B, 3, H, W).
        """
        # De acordo com a seção III-D, o fluxo é:
        # TFM1 -> D1 -> TFM2 -> UL1 -> D2 -> TFM3 -> UL2 -> D3
        
        x = self.tfm(cf, p1[0], p1[1]) # Aplica TFM1
        x = self.relu(self.d1(x))      # Aplica D1
        
        # Nível Intermediário (64 canais)
        x = self.tfm(x, p2[0], p2[1]) # Aplica TFM2
        x = self.upsample_x2(x)       # UL1
        x = self.relu(self.d2(x))     # Aplica D2
        
        # Nível Inicial (32 canais)
        x = self.tfm(x, p3[0], p3[1]) # Aplica TFM3
        x = self.upsample_x2(x)       # UL2
        x = self.d3(x)                # Aplica D3 (Saída não tem ReLU)
        
        return x