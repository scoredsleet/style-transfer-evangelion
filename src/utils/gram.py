import torch

def calc_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Calcula a Matriz de Gram para extrair a distribuição de estilo (correlação entre canais).

    Args:
        features (torch.Tensor): Tensor contendo os mapas de características
            extraídos por uma rede neural (ex: saídas de uma camada da VGG).
            O formato esperado do tensor é 4D: (B, C, H, W), onde:
            - B (Batch size): Quantidade de imagens sendo processadas simultaneamente.
            - C (Channels): Número de filtros/canais de características.
            - H (Height): Altura do mapa de características.
            - W (Width): Largura do mapa de características.

    Returns:
        torch.Tensor: A Matriz de Gram normalizada.
            O formato do tensor retornado é 3D: (B, C, C).
            Cada matriz C x C no batch representa o produto escalar (correlação) 
            entre todos os 'C' canais da imagem correspondente.
    """
    B, C, H, W = features.size()
    
    # 1. Achatamos a imagem fundindo Altura e Largura [B, C, H * W]. 
    # O foco aqui não é "onde" o traço do anime está, mas "qual" é a sua cor/textura.
    features_flat = features.view(B, C, H * W)
    
    # 2. Produto escalar entre os canais (features_flat) e sua transposta.
    # torch.bmm faz a multiplicação de matrizes em lote (batch matrix-matrix product).
    # (B, C, H*W) multiplicado por (B, H*W, C) resulta em (B, C, C).
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
    
    # 3. Normalizamos dividindo pelo número total de elementos para evitar que
    # matrizes de imagens maiores explodam os valores da rede durante o treino.
    gram = gram / (C * H * W)
    
    return gram
    
