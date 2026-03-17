import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class LossNetwork(nn.Module):
    """
    Rede VGG19 estática para cálculo de Perceptual e Style Loss.
    Os pesos são congelados e atuarão apenas como extratores de características.
    """

    def __init__(self):
        super(LossNetwork, self).__init__()

        # Carrega a VGG19 pré-treinada no ImageNet
        vgg_features = vgg19(weights=VGG19_Weights.DEFAULT).features

        # Converte os blocos da VGG para uma lista nativa do Python
        # Isso resolve o erro "Método __getitem__ não definido no tipo Module"
        features_list = list(vgg_features.children())

        # Fatiamos a lista nas camadas ReLU exigidas pelo artigo
        self.slice1 = nn.Sequential(*features_list[:4])   # Saída: relu1_2
        self.slice2 = nn.Sequential(*features_list[4:9])  # Saída: relu2_2
        self.slice3 = nn.Sequential(*features_list[9:18]) # Saída: relu3_4
        self.slice4 = nn.Sequential(*features_list[18:27])# Saída: relu4_4

        # Congela TODOS os parâmetros. Esta rede é a juíza, ela não aprende nada.
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, x: torch.Tensor):
        # Normalização ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        # Extrai as características em cada nível
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        # Retorna uma tupla com os 4 níveis de features
        return h1, h2, h3, h4 

class LCCLoss(nn.Module):
    """
    Agrupa as funções matemáticas para calcular o erro da rede LCCStyle.
    """
    def __init__(self, w_c=1.0, w_s=7.0, w_d=1.0):
        super(LCCLoss, self).__init__()
        self.w_c = w_c
        self.w_s = w_s
        self.w_d = w_d

    def calc_content_loss(self, gen_feat: torch.Tensor, cont_feat: torch.Tensor) -> torch.Tensor:
        """Calcula a perda de conteúdo (MSE) na camada relu3_4."""
        return F.mse_loss(gen_feat, cont_feat)

    def _calc_mean_std(self, feat: torch.Tensor, eps: float = 1e-5):
        """Função auxiliar para calcular a média e o desvio padrão de um mapa de características."""
        B, C, H, W = feat.size()
        feat_view = feat.view(B, C, -1)
        
        feat_mean = feat_view.mean(dim=2)
        feat_var = feat_view.var(dim=2) + eps
        feat_std = feat_var.sqrt()
        
        return feat_mean, feat_std

    def calc_style_loss(self, gen_feats: tuple, style_feats: tuple) -> torch.Tensor:
        """Calcula a perda de estilo alinhando as estatísticas em múltiplos níveis."""
        style_loss = 0.0
        for gen, styl in zip(gen_feats, style_feats):
            mu_g, sig_g = self._calc_mean_std(gen)
            mu_s, sig_s = self._calc_mean_std(styl)
            style_loss += F.mse_loss(mu_g, mu_s) + F.mse_loss(sig_g, sig_s)
        return style_loss

    def calc_disturbance_loss(self, img: torch.Tensor) -> torch.Tensor:
        """Calcula a Perda de Distúrbio baseada na diferença de brilho entre pixels vizinhos."""
        diff_h = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        diff_w = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        return diff_h.mean() + diff_w.mean()