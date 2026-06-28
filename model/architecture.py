import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    """Bloque básico de convolución para el Encoder y Decoder con SE y Residual"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SEBlock(out_channels)
        
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        self.final_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = self.residual(x)
        out = self.conv(x)
        out = self.se(out)
        out += res
        return self.final_relu(out)

class RNNBottleneck(nn.Module):
    def __init__(self, channels, freq_bins, hidden_size=256):
        super().__init__()
        self.in_features = channels * freq_bins
        self.fc_in = nn.Linear(self.in_features, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, self.in_features)
    
    def forward(self, x):
        # x shape: (B, C, F, T) -> (B, 256, 32, 33)
        b, c, f, t = x.size()
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f) # (B, 33, 8192)
        h = self.fc_in(x_reshaped)
        h, _ = self.lstm(h)
        out = self.fc_out(h)
        out = out.view(b, t, c, f).permute(0, 2, 3, 1).contiguous() # (B, C, F, T)
        return out + x # Conexión residual


class TinyUNetMultiStem(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (Contracción)
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool2d(2)
        
        # Cuello de Botella
        self.bottleneck = DoubleConv(128, 256)
        self.rnn_bottleneck = RNNBottleneck(channels=256, freq_bins=32, hidden_size=256)
        self.dropout = nn.Dropout2d(0.15)
        
        # Decoder (Expansión)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256, 128) # 128 (upconv) + 128 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        
        # Capa de Salida: 4 canales (Voces, Batería, Bajo, Otros)
        self.out_conv = nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))                    # Sin dropout en capas tempranas
        e3 = self.enc3(self.dropout(self.pool2(e2)))       # Dropout solo desde enc3
        e4 = self.enc4(self.dropout(self.pool3(e3)))
        
        # Bottleneck
        b = self.bottleneck(self.dropout(self.pool4(e4)))
        b = self.rnn_bottleneck(b)
        b = self.dropout(b)
        
        # Decoder con Skip Connections
        d4 = self.upconv4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)  # Regularización en decoder profundo
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Máscaras
        logits = self.out_conv(d1)
        
        # Sigmoid: cada máscara es independiente en [0, 1].
        # Permite solapamiento entre stems (más realista para audio).
        masks = torch.sigmoid(logits)
        return masks