import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importações do nosso projeto
from src.models.lccstyle import LCCStyleNet
from src.models.loss_network import LossNetwork, LCCLoss 
from src.utils.dataset import StyleTransferDataset

class LCCStyleTrainer:
    """
    Classe orquestradora para o treinamento da rede LCCStyle.
    Gerencia o otimizador, as funções de perda, o loop de épocas e o salvamento de checkpoints.
    """

    def __init__(self, 
                 model: LCCStyleNet, 
                 device: torch.device,
                 lr: float = 0.0001,
                 weight_content: float = 1.0,
                 weight_style: float = 7.0,
                 weight_disturbance: float = 1.0,
                 save_dir: str = "../data/weights"):
        """
        Inicializa os parâmetros de treinamento conforme o artigo LCCStyle.

        Args:
            model (LCCStyleNet): A rede principal a ser treinada.
            device (torch.device): CPU ou CUDA.
            lr (float): Taxa de aprendizado. O artigo fixa em 1e-4.
            weight_content (float): Peso da Perda de Conteúdo (preservação da forma).
            weight_style (float): Peso da Perda de Estilo (transferência de textura/cor).
            weight_disturbance (float): Peso da Perda de Distúrbio (evita mínimos locais).
            save_dir (str): Diretório para salvar os pesos do modelo (.pth).
        """
        self.device = device
        self.model = model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        if torch.cuda.device_count() > 1:
            print(f"🚀 Turbinando o treinamento: Usando {torch.cuda.device_count()} GPUs simultâneas!")
            self.model = torch.nn.DataParallel(self.model)

        # Configuração da Rede de Perdas (A VGG19 atua como juíza, não é treinada)
        self.loss_net = LossNetwork().to(self.device)
        self.loss_net.eval() # Sempre em modo de avaliação
        
        # Módulo que contém a matemática das perdas
        self.criterion = LCCLoss(w_c=weight_content, w_s=weight_style, w_d=weight_disturbance).to(self.device)

        # Otimizador: O artigo especifica o Adam
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Executa uma época completa de treinamento.
        """
        self.model.train()
        epoch_loss = 0.0
        
        # Barra de progresso para acompanhar visualmente
        pbar = tqdm(dataloader, desc=f"Época {epoch}", leave=False)

        for batch_idx, (content_img, style_img) in enumerate(pbar):
            # 1. Move os dados para a CPU/GPU
            content_img = content_img.to(self.device)
            style_img = style_img.to(self.device)

            # 2. Zera os gradientes acumulados da iteração anterior
            self.optimizer.zero_grad()

            # 3. Forward Pass: Gera a imagem estilizada
            generated_img = self.model(content_img, style_img)

            # 4. Extração de características pela VGG19 (O "Julgamento")
            gen_feats = self.loss_net(generated_img)
            cont_feats = self.loss_net(content_img)
            styl_feats = self.loss_net(style_img)

            # 5. Cálculo das Perdas Individuais
            # O artigo usa a relu3_4 (índice 2 na nossa LossNetwork) para conteúdo
            c_loss = self.criterion.calc_content_loss(gen_feats[2], cont_feats[2])
            s_loss = self.criterion.calc_style_loss(gen_feats, styl_feats)
            d_loss = self.criterion.calc_disturbance_loss(generated_img)

            # 6. Combinação Ponderada da Perda Total
            total_loss = (self.criterion.w_c * c_loss) + \
                         (self.criterion.w_s * s_loss) + \
                         (self.criterion.w_d * d_loss)

            # 7. Backward Pass: Calcula os gradientes
            total_loss.backward()

            # 8. Otimização: Atualiza os pesos da rede (Encoders, LTM, Decoder)
            self.optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        return epoch_loss / len(dataloader)

    def train(self, dataloader: DataLoader, epochs: int):
        """
        Loop principal de treinamento que salva apenas os melhores pesos.
        """
        print(f"Iniciando treinamento por {epochs} épocas no dispositivo: {self.device}")
        
        # 1. Inicializa a "melhor loss" com infinito. 
        # Assim, qualquer valor na primeira época já será considerado um recorde.
        best_loss = float('inf')
        
        # 2. Define o nome fixo do arquivo final
        best_model_path = os.path.join(self.save_dir, "best_lccstyle.pth")

        for epoch in range(1, epochs + 1):
            # Roda a época inteira e pega a média de erro
            avg_loss = self.train_epoch(dataloader, epoch)
            print(f"Época [{epoch}/{epochs}] - Loss Média: {avg_loss:.4f}")
            
            # 3. A Lógica de Checkpoint Inteligente
            if avg_loss < best_loss:
                print(f"🌟 Novo recorde! Loss caiu de {best_loss:.4f} para {avg_loss:.4f}. Salvando pesos...")
                best_loss = avg_loss
                # Salva (ou sobrescreve) o arquivo com os melhores pesos até agora
                model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                torch.save(model_to_save.state_dict(), best_model_path)
            
        print("="*50)
        print(f"Treinamento finalizado! O seu melhor modelo de Evangelion está salvo em:")
        print(f"-> {best_model_path}")
        print("="*50)
