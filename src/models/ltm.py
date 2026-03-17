import torch
import torch.nn as nn
from typing import Tuple


from ..utils.gram import calc_gram_matrix

class LTM(nn.Module):
    """
    Learning Transformation Module (LTM) baseado no algoritmo LCCStyle.

    Este módulo atua como uma hyper-network que utiliza convoluções 1D para
    processar a Matriz de Gram de uma imagem de estilo e gerar parâmetros 
    dinâmicos (W e b) para o módulo TFM.

    Atributos:
        in_channels (int): Número de canais do mapa de características de estilo
            (corresponde à dimensão da Matriz de Gram).
        tfm_channels (int): Número de canais de saída desejados para a camada 
            do TFM que receberá esses parâmetros.
    """

    def __init__(self, in_channels: int, tfm_channels: int):
        """
        Inicializa o LTM com convoluções 1D de baixa complexidade.

        Args:
            in_channels (int): Canais de entrada (ex: 128, 256 ou 512).
            tfm_channels (int): Canais de destino para a transformação.
        """
        super(LTM, self).__init__()

        self.in_channels = in_channels
        self.tfm_channels = tfm_channels
        
        # O artigo define o uso de Conv1d com kernel size 1 e stride 1 para 
        # minimizar o custo computacional e o número de parâmetros[cite: 333, 334].
        
        # Ramo para gerar os Pesos (Weights) da transformação
        self.conv_w = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=tfm_channels * 9, 
            kernel_size=1, # Valor definido no artigo
            stride=1       # Valor definido no artigo
        )
        
        # Ramo para gerar os Vieses (Biases) da transformação
        self.conv_b = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=tfm_channels, 
            kernel_size=1, # Valor definido no artigo
            stride=1       # Valor definido no artigo
        )

    def forward(self, style_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gera os parâmetros de estilo a partir das características extraídas.

        Args:
            style_features (torch.Tensor): Tensor 4D de características de estilo 
                no formato (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Um par de tensores (weight, bias) 
                no formato (B, tfm_channels, 1, 1), prontos para serem injetados 
                no módulo TFM[cite: 201, 246, 247].
        """
        gram = calc_gram_matrix(style_features) # Shape (B, C, C)
        B = gram.size(0)
        
        # Gera os pesos brutos: Shape (B, tfm_channels * 9, in_channels)
        weight_raw = self.conv_w(gram) 
        
        # Transforma os pesos para o formato exigido pelo PyTorch para Conv2d:
        # (Batch, Canais_Saida, Canais_Entrada, Kernel_H, Kernel_W)
        weight = weight_raw.view(B, self.tfm_channels, self.in_channels, 3, 3)
        
        # Gera os vieses: Shape (B, tfm_channels, in_channels)
        bias_raw = self.conv_b(gram)
        # O viés da convolução precisa ser apenas (B, tfm_channels).
        # Condensamos a última dimensão tirando a média das correlações.
        bias = bias_raw.mean(dim=2) 
        
        return weight, bias