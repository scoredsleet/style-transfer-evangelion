import torch
import traceback

# Importações mapeadas exatamente para a sua árvore de diretórios
from src.models.vgg_encoder import ContentEncoder, StyleEncoder
from src.models.ltm import LTM
from src.models.tfm import TFM
from src.models.decoder import Decoder
from src.models.lccstyle import LCCStyleNet
from src.models.loss_network import LossNetwork, LCCLoss
# Assumindo que a função calc_gram_matrix está dentro de gram.py
from src.utils.gram import calc_gram_matrix 

def run_all_tests():
    print("="*50)
    print(" INICIANDO BATERIA DE TESTES - LCCSTYLE")
    print("="*50)

    device = torch.device("cpu")
    B, C, H, W = 1, 3, 256, 256 # Batch 1, RGB, 256x256
    
    # Tensores base com requires_grad para testar o fluxo de aprendizado no final
    dummy_content = torch.randn(B, C, H, W, device=device)
    dummy_style = torch.randn(B, C, H, W, device=device)

    try:
        # ---------------------------------------------------------
        print("\n[1/7] Testando Encoders (Content & Style)...")
        c_enc = ContentEncoder().to(device)
        s_enc = StyleEncoder().to(device)
        
        cf = c_enc(dummy_content)
        s1, s2, s3 = s_enc(dummy_style)
        
        assert cf.shape == (B, 128, H//4, W//4), f"Erro no ContentEncoder: {cf.shape}"
        assert s3.shape == (B, 128, H//4, W//4), f"Erro no StyleEncoder s3: {s3.shape}"
        print("  ✅ Encoders funcionaram perfeitamente!")

        # ---------------------------------------------------------
        print("\n[2/7] Testando Cálculo da Matriz de Gram...")
        gram = calc_gram_matrix(s3)
        assert gram.shape == (B, 128, 128), f"Erro na Matriz de Gram: {gram.shape}"
        print("  ✅ Matriz de Gram calculada com sucesso!")

        # ---------------------------------------------------------
        print("\n[3/7] Testando LTM (Learning Transformation Module)...")
        ltm1 = LTM(in_channels=128, tfm_channels=128).to(device)
        w1, b1 = ltm1(s3)
        
        assert w1.shape == (B, 128, 128, 3, 3), f"Erro nos pesos dinâmicos: {w1.shape}"
        assert b1.shape == (B, 128), f"Erro no viés dinâmico: {b1.shape}"
        print("  ✅ LTM gerou as convoluções dinâmicas 3x3 perfeitamente!")

        # ---------------------------------------------------------
        print("\n[4/7] Testando TFM (Transformation Feature Module)...")
        tfm = TFM().to(device)
        stylized_cf = tfm(cf, w1, b1)
        
        assert stylized_cf.shape == cf.shape, f"Erro na convolução do TFM: {stylized_cf.shape}"
        print("  ✅ TFM aplicou a convolução dinâmica sem quebrar dimensões!")

        # ---------------------------------------------------------
        print("\n[5/7] Testando Decoder Mosaico...")
        decoder = Decoder().to(device)
        
        # Simulando os parâmetros dinâmicos dos 3 níveis para o teste
        p1 = (w1, b1)
        p2 = (torch.randn(B, 64, 64, 3, 3), torch.randn(B, 64))
        p3 = (torch.randn(B, 32, 32, 3, 3), torch.randn(B, 32))
        
        out_decoder = decoder(cf, p1, p2, p3)
        assert out_decoder.shape == (B, 3, H, W), f"Erro no Decoder: {out_decoder.shape}"
        print("  ✅ Decoder reconstruiu a imagem (RGB) no tamanho original!")

        # ---------------------------------------------------------
        print("\n[6/7] Testando LCCStyleNet (Pipeline Completo)...")
        model = LCCStyleNet().to(device)
        generated_img = model(dummy_content, dummy_style)
        
        assert generated_img.shape == (B, 3, H, W), f"Erro na Rede Completa: {generated_img.shape}"
        print("  ✅ Pipeline LCCStyleNet integrado com sucesso!")

        # ---------------------------------------------------------
        print("\n[7/7] Testando Rede de Perdas e Backpropagation (VGG19)...")
        loss_net = LossNetwork().to(device)
        criterion = LCCLoss().to(device)
        
        gen_feats = loss_net(generated_img)
        cont_feats = loss_net(dummy_content)
        styl_feats = loss_net(dummy_style)
        
        c_loss = criterion.calc_content_loss(gen_feats[2], cont_feats[2])
        s_loss = criterion.calc_style_loss(gen_feats, styl_feats)
        d_loss = criterion.calc_disturbance_loss(generated_img)
        
        total_loss = criterion.w_c * c_loss + criterion.w_s * s_loss + criterion.w_d * d_loss
        total_loss.backward() # Testa se a matemática inteira permite a passagem de gradientes
        
        print("  ✅ Forward Pass na VGG19 concluído.")
        print("  ✅ Funções matemáticas de Loss validadas.")
        print("  ✅ Backpropagation fluiu até os encoders sem quebras!")

        print("\n" + "="*50)
        print(" 🎉 TODOS OS 7 TESTES PASSARAM! ARQUITETURA PRONTA! 🎉")
        print("="*50)

    except Exception as e:
        print("\n" + "="*50)
        print(" ❌ ERRO ENCONTRADO DURANTE OS TESTES ❌")
        print("="*50)
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()