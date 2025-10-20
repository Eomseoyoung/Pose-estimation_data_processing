import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Transformer 모델을 위한 위치 인코딩.
    입력 시퀀스에 각 토큰의 위치 정보를 추가합니다.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model) 형태의 텐서
        Returns:
            위치 정보가 추가된 텐서
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PoseClassifierTransformer(nn.Module):
    """
    포즈 시퀀스 분류를 위한 경량 Transformer 모델.
    """
    def __init__(self,
                 input_dim: int = 12,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 128,
                 num_classes: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            input_dim (int): 입력 특징의 차원 (6개 관절 * 2D = 12).
            d_model (int): Transformer 모델의 내부 차원.
            nhead (int): Multi-head Attention의 헤드 수.
            num_encoder_layers (int): Transformer Encoder 레이어의 수.
            dim_feedforward (int): Feed-forward 네트워크의 은닉층 차원.
            num_classes (int): 분류할 클래스의 수 (3개 상태).
            dropout (float): Dropout 확률.
        """
        super().__init__()
        self.d_model = d_model

        # 1. 입력 임베딩 레이어
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 2. 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # 4. 분류기 헤드
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): (batch_size, seq_len, input_dim) 형태의 입력 텐서.
                                예: (32, 30, 12)
        Returns:
            torch.Tensor: (batch_size, num_classes) 형태의 클래스별 로짓(logits).
        """
        # 1. 입력 임베딩
        # src: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, d_model)
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)

        # 2. 위치 인코딩
        # PyTorch Transformer는 (seq_len, batch_size, d_model) 입력을 기대하므로 차원 변경
        pos_encoded_src = self.pos_encoder(embedded_src.permute(1, 0, 2))

        # 3. Transformer Encoder 처리
        # batch_first=True를 사용했으므로 다시 (batch_size, seq_len, d_model)로 변경
        # 하지만, nn.TransformerEncoder는 batch_first를 직접 지원하지 않으므로 permute 유지
        # output: (seq_len, batch_size, d_model)
        transformer_output = self.transformer_encoder(pos_encoded_src)

        # 4. 분류
        # 시퀀스의 모든 프레임 정보를 종합하기 위해 평균을 계산 (mean pooling)
        # output: (seq_len, batch_size, d_model) -> (batch_size, d_model)
        pooled_output = transformer_output.mean(dim=0)

        # 최종 분류 레이어를 통과하여 로짓(logits) 생성
        # logits: (batch_size, d_model) -> (batch_size, num_classes)
        logits = self.classifier(pooled_output)

        return logits