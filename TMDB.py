import requests
import os

# 1. Coloque sua chave de API do TMDB aqui
API_KEY = '<>'
SERIES_ID = ID # ID de Neon Genesis Evangelion

# Endpoint da API para pegar imagens de uma série de TV
#url_api = f"https://api.themoviedb.org/3/tv/{SERIES_ID}/images?api_key={API_KEY}"
url_api = f"https://api.themoviedb.org/3/movie/{SERIES_ID}/images?api_key={API_KEY}"

def baixar_backdrops_api():
    resposta = requests.get(url_api)
    dados = resposta.json()

    # O TMDB retorna uma lista de 'backdrops', 'posters' e 'logos'
    backdrops = dados.get('backdrops', [])
    
    pasta_destino = "images"
    os.makedirs(pasta_destino, exist_ok=True)

    print(f"Encontrados {len(backdrops)} backdrops. Iniciando o download...")

    for index, imagem in enumerate(backdrops):
        # O file_path vem apenas como "/caminho_da_imagem.jpg"
        file_path = imagem['file_path']
        
        # O TMDB permite escolher o tamanho. 'original' pega a qualidade máxima.
        url_imagem_completa = f"https://image.tmdb.org/t/p/original{file_path}"
        
        # Baixando a imagem
        img_data = requests.get(url_imagem_completa).content
        nome_arquivo = os.path.join(pasta_destino, f"evangelion_3_33_{index + 1}.jpg")
        
        with open(nome_arquivo, 'wb') as handler:
            handler.write(img_data)
            
        print(f"Baixado: {nome_arquivo}")

def alterar_nome():
    pasta_destino = "images" 

    print("Iniciando a renomeação dos arquivos...")

    # 2. Cria um loop que vai do número 1 até o 100
    # Nota: no Python, o range(1, 101) para no 100.
    for i in range(1, 101):
        # 3. Define como o nome era e como deve ficar
        nome_antigo = f"eva_3_0+1_backdrop_{i}.jpg"
        nome_novo = f"evangelion_3_0+1_0_{i}.jpg"

        # 4. Junta o nome da pasta com o nome do arquivo
        caminho_antigo = os.path.join(pasta_destino, nome_antigo)
        caminho_novo = os.path.join(pasta_destino, nome_novo)

        # 5. Verifica se o arquivo antigo realmente existe antes de tentar renomear
        if os.path.exists(caminho_antigo):
            os.rename(caminho_antigo, caminho_novo)
            print(f"Sucesso: {nome_antigo} -> {nome_novo}")
        else:
            print(f"Aviso: O arquivo {nome_antigo} não foi encontrado na pasta.")

        print("Processo finalizado!")

# Descomente a linha abaixo para rodar se tiver a API Key
baixar_backdrops_api()
# alterar_nome()