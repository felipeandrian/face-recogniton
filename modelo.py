import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconhecedorFacialCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(ReconhecedorFacialCNN, self).__init__()
        
        # Bloco 1: Detecta bordas e texturas simples
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # O Estabilizador
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloco 2: Detecta formatos (olhos, nariz)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloco 3: Detecta o rosto complexo
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculando o tamanho após 3 pools de tamanho 2x2:
        # Imagem entra 224x224 -> Pool1(112) -> Pool2(56) -> Pool3(28)
        # 128 canais * 28 * 28 = 100352
        
        # O "Cérebro" Final
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5) # O Antidecoreba (Desliga 50% dos neurônios no treino)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Passando a imagem pelos blocos
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Achatando a imagem para entrar na camada Linear
        x = x.view(-1, 128 * 28 * 28)
        
        # Decisão
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Aplica o Dropout antes da decisão final
        x = self.fc2(x)
        
        return x
