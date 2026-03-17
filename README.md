# Evangelion LCCStyle Transfer

Este projeto implementa uma arquitetura de Transferência de Estilo Arbitrária com Baixa Complexidade Computacional (LCCStyle), focada em aplicar a estética do anime Neon Genesis Evangelion em imagens do mundo real.

O objetivo é processar inferências rapidamente em CPU/GPU através da substituição de camadas densas por Convoluções 1D no módulo LTM.

## 🚀 Roadmap de Desenvolvimento

- [ ] **Setup Inicial:** Estruturação de diretórios e classes base.
- [ ] **Módulo 1 (Encoder):** Importar VGG-19 e congelar pesos para extração de *features*.
- [ ] **Módulo 2 (LTM):** Implementar o *Learning Transformation Module* com Conv1D.
- [ ] **Módulo 3 (TFM):** Implementar o *Transformation Feature Module* para fusão de características.
- [ ] **Módulo 4 (Decoder):** Criar a rede de reconstrução da imagem final.
- [ ] **Loss Functions:** Implementar Content Loss e Style Loss.
- [ ] **Utilitários de Apresentação:** Classes para plotar comparações (AdaIN vs LCCStyle vs Original).
- [ ] **Treinamento:** Pipeline de *fine-tuning* com 450 imagens de estilo de Evangelion.

## 🛠️ Requisitos
Instale as dependências executando:
\`\`\`bash
pip install -r requirements.txt
\`\`\`