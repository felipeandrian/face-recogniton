import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from modelo import ReconhecedorFacialCNN # Importa a classe da arquitetura (o "cérebro" vazio)

# ==========================================
# 1. CONFIGURAÇÕES DE AMBIENTE
# ==========================================
data_dir = 'dataset_cientistas'

# Detecta automaticamente se o seu computador tem uma placa de vídeo da NVIDIA (CUDA).
# Se tiver, o treino será até 50x mais rápido. Se não, ele usa o processador normal (CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Iniciando Treinamento Longo em: {device} ---")

# ==========================================
# 2. DATA AUGMENTATION (O SEGREDO CONTRA O OVERFITTING)
# ==========================================
# Se a IA ver as mesmas fotos do mesmo jeito sempre, ela "decora" a imagem (Overfitting).
# O Data Augmentation altera as fotos sutilmente a cada época, forçando a IA a aprender o ROSTO real.
transformacoes = {
    'treino': transforms.Compose([
        transforms.Resize((224, 224)), # Força todas as imagens a terem o mesmo tamanho quadrado
        transforms.RandomHorizontalFlip(p=0.5), # 50% de chance de espelhar a foto (ensina a ver o rosto de ambos os lados)
        transforms.RandomRotation(15), # Gira a foto em até 15 graus (ajuda com fotos tortas)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Move o rosto um pouco para os lados/cima/baixo
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Altera o brilho e contraste (simula luzes diferentes)
        transforms.ToTensor(), # Transforma a imagem (pixels) em tensores matemáticos (matrizes) para o PyTorch
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normaliza as cores para o padrão universal ImageNet
    ]),
    
    # Na validação (a "prova" final da IA), NÃO fazemos bagunça. Apenas ajustamos o tamanho e normalizamos.
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 3. CARREGAR DATASETS E DATALOADERS
# ==========================================
# O ImageFolder é mágico: ele entra na pasta 'treino' e usa o NOME das subpastas (ex: 'elon_musk') como a resposta certa (label).
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transformacoes[x])
                  for x in ['treino', 'val']}

# O DataLoader pega as milhares de fotos e as agrupa em "lotes" (batches) de 32 imagens por vez.
# Isso impede que o computador fique sem memória RAM e ajuda a IA a aprender de forma mais estável.
# shuffle=True embaralha as fotos para que a IA não aprenda a ordem em que elas aparecem.
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['treino', 'val']}

class_names = image_datasets['treino'].classes
num_classes = len(class_names) # Descobre automaticamente quantas pessoas tem no dataset (ex: 4 classes)

# ==========================================
# 4. INSTANCIAR O MODELO
# ==========================================
# Cria o corpo da sua rede neural e envia ele para a memória da CPU ou GPU
model = ReconhecedorFacialCNN(num_classes=num_classes).to(device)

# ==========================================
# 5. OTIMIZADOR, FUNÇÃO DE PERDA E SCHEDULER
# ==========================================
# CrossEntropyLoss: É a métrica usada para classificar categorias. Ela pune severamente a IA se ela errar com muita confiança.
criterion = nn.CrossEntropyLoss()

# Adam: É o algoritmo matemático que vai atualizar os "pesos" (o conhecimento) da rede.
# lr (Learning Rate): A velocidade do aprendizado. Um valor muito alto faz a IA "pular" a resposta certa; muito baixo demora demais.
# weight_decay (L2 Penalty): Força os pesos da rede a ficarem pequenos, dificultando que ela decore pixels específicos.
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

# Scheduler (O Freio): A cada 30 épocas (step_size), ele multiplica a velocidade de aprendizado por 0.1 (gamma).
# Ou seja, o aprendizado vai ficando cada vez mais lento e preciso (ajuste fino) conforme o tempo passa.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# ==========================================
# 6. LOOP DE TREINAMENTO (A SALA DE AULA)
# ==========================================
num_epochs = 100
melhor_acc = 0.0 # Variável para guardar o recorde de acurácia

for epoch in range(num_epochs):
    print(f'Época {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Cada época tem duas fases: Estudar (treino) e Fazer a Prova (validação)
    for phase in ['treino', 'val']:
        if phase == 'treino':
            model.train() # Avisa camadas como Dropout e BatchNorm para ATIVAREM suas funções de treino
        else:
            model.eval()  # Avisa essas mesmas camadas para DESATIVAREM (comportamento de inferência/produção)

        running_loss = 0.0
        running_corrects = 0

        # Percorre as imagens lote por lote (32 por vez)
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zera a memória de erros do otimizador da etapa anterior (obrigatório no PyTorch)
            optimizer.zero_grad()

            # Só calcula os gradientes (matemática pesada de aprendizado) se estiver na fase de TREINO
            with torch.set_grad_enabled(phase == 'treino'):
                outputs = model(inputs) # O modelo tenta adivinhar quem está na foto
                _, preds = torch.max(outputs, 1) # Pega o palpite com a maior probabilidade
                loss = criterion(outputs, labels) # Calcula a diferença (erro) entre o palpite e a resposta real

                # Se estiver treinando, faz a "Retropropagação" (Backpropagation)
                if phase == 'treino':
                    loss.backward()  # Calcula onde a rede errou em cada camada
                    optimizer.step() # Ajusta os parafusos (pesos) para não errar de novo

            # Soma os erros e os acertos deste lote para as estatísticas finais da época
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Se a fase for treino, dá um passo no Scheduler (se for a época 30, 60 ou 90, ele aciona o freio)
        if phase == 'treino':
            scheduler.step()

        # Calcula a média de erros e acertos da época inteira
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # ==========================================
        # 7. SISTEMA DE SALVAMENTO (CHECKPOINT)
        # ==========================================
        # Se estivermos na "prova" (validação) e a IA tirou uma nota MAIOR que o recorde anterior:
        if phase == 'val' and epoch_acc > melhor_acc:
            melhor_acc = epoch_acc # Atualiza o recorde
            # Salva o "cérebro" (.pth). Isso garante que, mesmo se a rede piorar nas últimas épocas,
            # nós teremos salvo o momento exato em que ela estava no seu auge de inteligência.
            torch.save(model.state_dict(), "modelo_cientistas.pth")

print(f'\nTreino finalizado! Melhor Acurácia de Validação: {melhor_acc:.4f}')
