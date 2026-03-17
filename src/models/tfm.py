
import torch
import torch.nn as nn
import torch.nn.functional as F

class TFM(nn.Module):
    """
    Transformation Feature Module (TFM) do algoritmo LCCStyle.

    Este módulo é responsável por transformar o mapa de características de conteúdo
    em um mapa estilizado utilizando os parâmetros dinâmicos gerados pelo LTM.
    """

    def __init__(self):
        """
        Inicializa o TFM. 
        Note que o TFM não possui parâmetros aprendíveis próprios (pesos fixos),
        pois ele consome os parâmetros gerados dinamicamente.
        """
        super(TFM, self).__init__()
        # De acordo com o artigo, o TFM é seguido por uma ativação ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, content_feature: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Aplica a transferência de estilo no domínio de características.

        Args:
            content_features (torch.Tensor): Mapa de características de conteúdo (B, C, H, W).
            weight (torch.Tensor): Pesos dinâmicos gerados pelo LTM (B, C, 1, 1).
            bias (torch.Tensor): Vieses dinâmicos gerados pelo LTM (B, C, 1, 1).

        Returns:
            torch.Tensor: Mapa de características estilizado.
        """

        B, C_in, H, W = content_feature.shape
        _, C_out, _, kH, kW = weight.shape
        
        # Para aplicar F.conv2d com pesos diferentes para CADA imagem no batch,
        # usamos o truque de "groups". Remodelamos a entrada e os pesos:
        
        # Junta todas as imagens do batch no eixo dos canais: Shape (1, B * C_in, H, W)
        x_reshaped = content_feature.view(1, B * C_in, H, W)
        
        # Junta todos os pesos do batch: Shape (B * C_out, C_in, 3, 3)
        w_reshaped = weight.view(B * C_out, C_in, kH, kW)
        
        # Junta os vieses: Shape (B * C_out)
        b_reshaped = bias.view(B * C_out)
        
        # Aplica a convolução com padding=1 para manter H e W intactos.
        # groups=B garante que a imagem 1 use os pesos 1, a imagem 2 use os pesos 2...
        out = F.conv2d(x_reshaped, w_reshaped, bias=b_reshaped, padding=1, groups=B)
        
        # Retorna ao formato original do batch: Shape (B, C_out, H, W)
        out = out.view(B, C_out, H, W)
        
        return self.relu(out)