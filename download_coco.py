import os
import random
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Barra de progresso customizada para o download do arquivo ZIP."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def prepare_coco_dataset(num_images=3000, target_dir="./data/content"):
    # URL oficial do MS COCO val2017 (Aproximadamente 1 GB)
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = "val2017.zip"
    extracted_folder = "val2017"

    # Garante que a pasta de destino existe
    os.makedirs(target_dir, exist_ok=True)

    # 1. Download
    if not os.path.exists(zip_path):
        print(f"Iniciando o download do MS COCO Val2017 (~1 GB)...")
        download_url(url, zip_path)
    else:
        print("Arquivo ZIP já encontrado localmente.")

    # 2. Extração
    print("\nExtraindo imagens do ZIP (Isso pode levar alguns segundos)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    # 3. Listagem e Sorteio Aleatório
    all_images = [f for f in os.listdir(extracted_folder) if f.endswith('.jpg')]
    print(f"Total de imagens encontradas: {len(all_images)}")
    
    # Previne erros caso peçamos mais imagens do que o dataset possui
    samples_to_take = min(num_images, len(all_images))
    selected_images = random.sample(all_images, samples_to_take)

    # 4. Movendo para a pasta do projeto
    print(f"\nSelecionando {samples_to_take} imagens aleatórias e movendo para {target_dir}...")
    for img_name in tqdm(selected_images, desc="Movendo imagens"):
        src = os.path.join(extracted_folder, img_name)
        dst = os.path.join(target_dir, img_name)
        shutil.move(src, dst)

    # 5. Limpeza (Evita lixo no seu disco)
    print("\nLimpando arquivos temporários (Apagando o ZIP e a pasta original)...")
    shutil.rmtree(extracted_folder)
    os.remove(zip_path)

    print("\n[SUCESSO] Dataset de Conteúdo pronto para o treinamento!")
    print(f"Você agora tem {samples_to_take} fotos reais na pasta '{target_dir}'.")

if __name__ == "__main__":
    # Executa a função pedindo 3000 imagens
    prepare_coco_dataset(num_images=3000)