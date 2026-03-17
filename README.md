
#  Custom CNN for Facial Recognition (Educational PoC)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-F8D521?style=for-the-badge)

Uma Prova de Conceito (Proof of Concept) desenvolvida para demonstrar os fundamentos de Deep Learning e Visão Computacional. Este projeto implementa uma Rede Neural Convolucional (CNN) **construída inteiramente do zero**, sem o uso de modelos pré-treinados (Transfer Learning), para a classificação e reconhecimento de personalidades históricas e contemporâneas.

 **[Live Demo no Hugging Face Spaces](https://felipeandrian-face-recognition-identities.hf.space/)**

---

##  Objetivos de Engenharia

Em ambientes de produção modernos, o reconhecimento facial é frequentemente resolvido via *Metric Learning* (ex: redes siamesas, ArcFace). O objetivo primário deste repositório, no entanto, é **educacional**: demonstrar o domínio sobre o fluxo completo de Machine Learning, desde a definição topológica da rede até o deploy da inferência.

* **Design Arquitetural:** Construção de uma CNN customizada otimizada para CPUs.
* **Mitigação de Overfitting:** Implementação de técnicas de regularização (Dropout, Weight Decay) para lidar com *datasets* com alta variância e baixo volume.
* **Generalização:** Uso de *Data Augmentation* dinâmico para garantir que a rede aprenda características faciais intrínsecas (features geométricas) e não artefatos de fundo (background bias).
* **Deploy:** Encapsulamento da inferência em uma interface web reativa via Gradio.

---

##  Topologia da Rede (Architecture)

A classe `ReconhecedorFacialCNN` herda de `torch.nn.Module` e foi projetada utilizando uma hierarquia clássica de extração de características em 3 blocos de profundidade espacial, culminando em uma rede densa para classificação.



| Camada (Layer) | Tipo | Configuração | Propósito |
| :--- | :--- | :--- | :--- |
| **Block 1** | Conv2d + BatchNorm + ReLU + MaxPool | 3 -> 32 canais, kernel 3x3 | Extração de bordas e texturas de baixo nível. |
| **Block 2** | Conv2d + BatchNorm + ReLU + MaxPool | 32 -> 64 canais, kernel 3x3 | Detecção de formas e contornos faciais (nariz, olhos). |
| **Block 3** | Conv2d + BatchNorm + ReLU + MaxPool | 64 -> 128 canais, kernel 3x3 | Composição de features de alto nível (rostos específicos). |
| **Classifier 1**| Linear + ReLU | 100352 -> 512 neurônios | Sintetização do vetor de características. |
| **Regularizer** | Dropout | $p = 0.5$ | Desativa 50% dos neurônios no treino para forçar generalização redundante. |
| **Classifier 2**| Linear (Output) | 512 -> N classes | Logits finais para a função Softmax. |

*Nota: A inclusão de `BatchNorm2d` foi crucial para estabilizar o gradiente durante a retropropagação (backpropagation), permitindo que a rede convergisse mais rapidamente mesmo sem inicialização de pesos de Xavier/Kaiming explícita.*

---

## Estratégia de Treinamento e Resultados

O treinamento foi conduzido sob o desafio de um conjunto de dados extramemnte restrito (poucas amostras por classe cerca de apensa 15 imagens). Para evitar que a rede memorizasse as imagens de treino, o seguinte pipeline foi adotado no `treino.py`:

1.  **Transformações Estocásticas (Augmentation):** `RandomHorizontalFlip`, `RandomRotation(15)`, `RandomAffine` e `ColorJitter`. Isso gerou um dataset virtualmente infinito, forçando a invariância rotacional e de iluminação.
2.  **Otimizador:** Adam com `learning_rate` inicial conservador de `3e-4` e penalidade L2 (`weight_decay=1e-4`).
3.  **Learning Rate Scheduling:** Utilização do `StepLR` reduzindo o passo em 10x a cada 30 épocas, permitindo que o otimizador fizesse ajustes finos no platô da função de perda (Loss).
4.  **Early Stopping (Manual):** O estado do dicionário (`state_dict`) só era salvo em disco quando a acurácia de validação superava o recorde histórico.

**Métricas Finais (Epoch 99):**
* **Validation Accuracy:** `97.35%`
* **Validation Loss:** `0.1216`

Para obter resultados realmente eficientes é necessario uma amostram muito maior do que a do exemplo, esse modelo foi apenas para demonstração de criação do zero.

---

##  Estrutura do Repositório

```bash
├── app.py                   # Script de inferência e interface web (Gradio)
├── modelo.py                # Definição da classe da arquitetura da CNN em PyTorch
├── treino.py                # Loop de treinamento principal com Data Augmentation e Scheduler
├── .gitignore               # Exclusão de datasets e pesos salvos localmente
└── README.md                # Documentação técnica

```

*(Arquivos `.pth` e o diretório `dataset_cientistas/` não estão versionados por boas práticas de gestão de repositórios de IA).*

---

## Como Executar Localmente

### Pré-requisitos

* Python 3.8+
* PyTorch, Torchvision, Gradio, Pillow

### Passos para Treinar

1. Clone o repositório:
```bash
git clone [https://github.com/felipeandrian/face-recogniton.git
cd SEU_REPOSITORIO

```


2. Crie a estrutura de pastas do dataset contendo suas próprias imagens:
```text
dataset_cientistas/
├── treino/
│   ├── classe_A/
│   └── classe_B/
└── val/
    ├── classe_A/
    └── classe_B/

```


3. Execute o script de treinamento:
```bash
python treino.py

```


*O melhor modelo será salvo como `modelo_cientistas.pth`.*

### Passos para Inferência (Interface Web)

Após gerar o arquivo `.pth` (ou se o tiver baixado separadamente), inicie o servidor local:

```bash
python app.py

```

O Gradio disponibilizará a interface no endereço `http://localhost:7860`.

---

##  Trabalhos Futuros

Para evoluir esta arquitetura para um cenário de larga escala, os seguintes passos seriam implementados:

* Substituição do `nn.Linear` final por **Global Average Pooling** (redução agressiva de parâmetros).
* Implementação de uma função de perda angular (ex: **CosFace** ou **ArcFace**) para forçar maior distância inter-classes.
* Migração do pipeline de treino para GPUs/TPUs utilizando PyTorch Lightning.

