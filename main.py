import os
import argparse
import torch
from torch.utils.data import DataLoader

# Importações do nosso pacote 'src'
from src.models.lccstyle import LCCStyleNet
from src.utils.dataset import StyleTransferDataset
from src.train import LCCStyleTrainer
from src.inference import run_inference

from src.test import run_all_tests

def print_banner():
    """Imprime um banner profissional no terminal."""
    banner = """
    =============================================================
     EVA-LCCStyle: Arbitrary Style Transfer (Neon Genesis Evangelion)
     Baseado no artigo LCCStyle (Low Computational Complexity)
    =============================================================
    """
    print(banner)

def setup_parser() -> argparse.ArgumentParser:
    """Configura o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo de Transferência de Estilo LCCStyle.",
        epilog="""
Exemplos de Uso:
  Treinamento: python main.py --mode train --epochs 15 --batch_size 2
  Inferência:  python main.py --mode infer --content foto.jpg --style eva.jpg
  Testes:      python main.py --mode test
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Argumento principal (Obrigatório)
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer', 'test'],
                        help="Modo de execução: 'train' (treinar), 'infer' (gerar imagem) ou 'test' (validar arquitetura).")

    # ==========================================
    # Grupo 1: Argumentos de Treinamento
    # ==========================================
    train_group = parser.add_argument_group('Configurações de Treinamento')
    train_group.add_argument('--content_dir', type=str, default='data/content', help="Pasta com as fotos reais (Ex: MS COCO).")
    train_group.add_argument('--style_dir', type=str, default='data/style', help="Pasta com os frames de Evangelion, todos os rebuilds e o anime de 1995.")
    train_group.add_argument('--weights_dir', type=str, default='data/weights', help="Pasta para salvar os pesos (best_lccstyle.pth).")
    train_group.add_argument('--epochs', type=int, default=10, help="Número total de épocas.")
    train_group.add_argument('--batch_size', type=int, default=4, help="Tamanho do batch.")
    train_group.add_argument('--lr', type=float, default=0.0001, help="Taxa de aprendizado (Artigo LCCStyle especifica 1e-4).")
    
    # Pesos das funções de perda (Losses)
    train_group.add_argument('--w_content', type=float, default=1.0, help="Peso da Perda de Conteúdo.")
    train_group.add_argument('--w_style', type=float, default=7.0, help="Peso da Perda de Estilo.")
    train_group.add_argument('--w_dist', type=float, default=1.0, help="Peso da Perda de Distúrbio.")

    # ==========================================
    # Grupo 2: Argumentos de Inferência
    # ==========================================
    infer_group = parser.add_argument_group('Configurações de Inferência')
    infer_group.add_argument('--content', type=str, default='data/content/test.jpg', help="Caminho da foto real a ser estilizada.")
    infer_group.add_argument('--style', type=str, default='data/style/eva01.jpg', help="Caminho da imagem de referência de estilo.")
    infer_group.add_argument('--output', type=str, default='output/out.png', help="Caminho para salvar a arte gerada.")
    infer_group.add_argument('--model_weights', type=str, default='data/weights/best_lccstyle.pth', help="Caminho dos pesos treinados.")

    return parser

def main():
    print_banner()
    parser = setup_parser()
    args = parser.parse_args()

    # Define dispositivo de hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # MODO 1: TESTES DE SANIDADE
    # ---------------------------------------------------------
    if args.mode == 'test':
        print(f"[*] Modo Selecionado: TESTES DE ARQUITETURA")
        run_all_tests()
        return

    # ---------------------------------------------------------
    # MODO 2: TREINAMENTO
    # ---------------------------------------------------------
    elif args.mode == 'train':
        print(f"[*] Modo Selecionado: TREINAMENTO (Device: {device})")
        
        # Valida se as pastas existem
        if not os.path.exists(args.content_dir) or not os.path.exists(args.style_dir):
            raise FileNotFoundError(f"As pastas de dataset não foram encontradas! Verifique: {args.content_dir} e {args.style_dir}")

        # Instancia modelo e dataset
        model = LCCStyleNet()
        dataset = StyleTransferDataset(content_dir=args.content_dir, style_dir=args.style_dir, image_size=300)
        
        if len(dataset) == 0:
            raise ValueError("O dataset está vazio! Execute o script de download do COCO ou adicione imagens.")

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Instancia o Trainer repassando todos os argumentos do CLI
        trainer = LCCStyleTrainer(
            model=model,
            device=device,
            lr=args.lr,
            weight_content=args.w_content,
            weight_style=args.w_style,
            weight_disturbance=args.w_dist,
            save_dir=args.weights_dir
        )

        # Inicia o loop (com o salvamento inteligente do 'best_loss')
        trainer.train(dataloader=dataloader, epochs=args.epochs)

    # ---------------------------------------------------------
    # MODO 3: INFERÊNCIA
    # ---------------------------------------------------------
    elif args.mode == 'infer':
        print(f"[*] Modo Selecionado: INFERÊNCIA (Geração de Imagem)")
        
        if not os.path.exists(args.content):
            raise FileNotFoundError(f"Imagem de conteúdo não encontrada: {args.content}")
        if not os.path.exists(args.style):
            raise FileNotFoundError(f"Imagem de estilo não encontrada: {args.style}")

        # Garante que a pasta output/ existe
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        run_inference(
            content_path=args.content,
            style_path=args.style,
            output_path=args.output,
            model_weights=args.model_weights
        )

if __name__ == "__main__":
    main()