import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, hidden_size, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(1000, embedding_size)  # 위치 인코딩
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, self.embedding_size).to(x.device)
        x = self.embedding(x) + self.position_embedding(positions)
        x = x.permute(1, 0, 2)  # Transformer에 맞게 (seq_len, batch_size, embedding_size)로 변환
        output = self.transformer_encoder(x)
        output = output.permute(1, 0, 2)  # 다시 (batch_size, seq_len, embedding_size)로 변환
        output = self.fc(output)
        return output