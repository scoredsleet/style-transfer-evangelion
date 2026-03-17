import torch
import cv2
import numpy as np
import os
from src.models.lccstyle import LCCStyleNet

def load_image_cv2(image_path: str, size: int = 512) -> torch.Tensor:
    """
    Carrega e prepara a imagem para a rede neural usando OpenCV.
    """
    # 1. Lê a imagem (retorna um array NumPy BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # 2. Converte de BGR para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Redimensiona para o tamanho desejado (ex: 512x512)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # 4. Converte para float32 e normaliza para o intervalo [0.0, 1.0]
    img = img.astype(np.float32) / 255.0

    # 5. Converte para Tensor PyTorch e ajusta as dimensões
    # De [H, W, C] para [C, H, W]
    tensor = torch.from_numpy(img).permute(2, 0, 1)

    # 6. Adiciona a dimensão do Batch: de [C, H, W] para [1, C, H, W]
    return tensor.unsqueeze(0)

def save_image_cv2(tensor: torch.Tensor, output_path: str):
    """
    Pós-processa o tensor de volta para uma imagem real e salva com OpenCV.
    """
    # 1. Remove a dimensão do Batch e move para a CPU (se estiver na GPU)
    img_tensor = tensor.detach().cpu().squeeze(0)
    
    # 2. Normalização Min-Max (conforme artigo LCCStyle) para evitar cortes de cor
    f_min = img_tensor.min()
    f_max = img_tensor.max()
    img_tensor = (img_tensor - f_min) / (f_max - f_min + 1e-5)

    # 3. Converte de Tensor para NumPy Array e ajusta as dimensões
    # De [C, H, W] de volta para [H, W, C]
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # 4. Desnormaliza (de 0.0~1.0 para 0~255) e converte para inteiros (uint8)
    img_np = (img_np * 255.0).astype(np.uint8)

    # 5. Converte de RGB de volta para BGR (formato que o OpenCV usa para salvar)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 6. Salva a imagem no disco
    cv2.imwrite(output_path, img_bgr)
    print(f"Imagem salva com sucesso em: {output_path}")

def run_inference(content_path: str, style_path: str, output_path: str, model_weights: str = None):
    # Configura o dispositivo (sua CPU ou GPU, se houver)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando transferência de estilo no dispositivo: {device}")

    # Inicializa a rede LCCStyle
    model = LCCStyleNet().to(device)
    
    # Carrega os pesos (se o modelo já foi treinado)
    if model_weights and os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights, map_location=device))
        print("Pesos do modelo carregados com sucesso.")
    else:
        print("Aviso: Rodando com pesos aleatórios não treinados. A saída será abstrata.")

    model.eval() # Modo de inferência

    # Carrega e processa as imagens via OpenCV
    content_tensor = load_image_cv2(content_path).to(device)
    style_tensor = load_image_cv2(style_path).to(device)

    # Faz o Forward Pass (aplica o estilo de Evangelion)
    print("Processando a imagem...")
    with torch.no_grad():
        stylized_tensor = model(content_tensor, style_tensor)

    # Salva o resultado
    save_image_cv2(stylized_tensor, output_path)