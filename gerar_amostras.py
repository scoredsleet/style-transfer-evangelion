import os
import random
import cv2
import matplotlib.pyplot as plt

# Importa a função de inferência que você já criou!
from src.inference import run_inference

def plot_side_by_side(content_path, style_path, result_path, amostra_num, output_dir):
    """Lê as três imagens do disco e plota lado a lado."""
    # Lê as imagens usando OpenCV
    img_c = cv2.imread(content_path)
    img_s = cv2.imread(style_path)
    img_r = cv2.imread(result_path)

    if img_c is None or img_s is None or img_r is None:
        print(f"Erro ao ler uma das imagens para o mosaico {amostra_num}")
        return

    # Converte de BGR para RGB
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Avaliação Aleatória - Amostra {amostra_num}', fontsize=16, fontweight='bold')

    axes[0].imshow(img_c)
    axes[0].set_title("1. Conteúdo", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(img_s)
    axes[1].set_title("2. Estilo", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(img_r)
    axes[2].set_title("3. Resultado LCCStyle", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    
    # CORREÇÃO: Salva o mosaico na pasta de output correta
    mosaico_nome = f'mosaico_amostra_{amostra_num}.jpg'
    mosaico_path = os.path.join(output_dir, mosaico_nome)
    
    plt.savefig(mosaico_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Mosaico salvo em: {mosaico_path}")

def gerar_amostras_aleatorias(num_amostras=5):
    # Pega o caminho de onde o script está sendo executado
    base_path = os.getcwd()
    
    # CONFIGURAÇÃO DE PASTAS (Ajuste esses nomes se as suas pastas tiverem nomes diferentes)
    content_dir = os.path.join(base_path, 'data', 'content')
    style_dir = os.path.join(base_path, 'data', 'style')
    output_dir = os.path.join(base_path, 'data', 'output')
    model_weights = os.path.join(base_path, 'data', 'weights', 'best_lccstyle_final.pth')

    # Cria a pasta de output se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Pasta criada: {output_dir}")

    # Verifica se os diretórios existem antes de tentar listar
    if not os.path.exists(content_dir) or not os.path.exists(style_dir):
        print(f"ERRO: Pasta de conteúdo ou estilo não encontrada!")
        print(f"Procurando em: {content_dir}")
        return

    formatos = ('.jpg', '.jpeg', '.png')
    content_files = [f for f in os.listdir(content_dir) if f.lower().endswith(formatos)]
    style_files = [f for f in os.listdir(style_dir) if f.lower().endswith(formatos)]

    if not content_files or not style_files:
        print("Erro: Pastas vazias!")
        return

    for i in range(1, num_amostras + 1):
        print(f"\n🚀 Processando Amostra [{i}/{num_amostras}]...")
        
        random_content = random.choice(content_files)
        random_style = random.choice(style_files)

        c_path = os.path.join(content_dir, random_content)
        s_path = os.path.join(style_dir, random_style)
        r_path = os.path.join(output_dir, f'resultado_aleatorio_{i}.jpg')

        # Tenta rodar a inferência
        try:
            run_inference(c_path, s_path, r_path, model_weights)
            plot_side_by_side(c_path, s_path, r_path, i, output_dir)
        except Exception as e:
            print(f"Erro na inferência da amostra {i}: {e}")

if __name__ == "__main__":
    gerar_amostras_aleatorias(num_amostras=3)