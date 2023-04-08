import math
import torch

from torch import nn


class SeqCls(nn.Module):
    '''
    基于LSTM的序列分类模型。
    '''

    def __init__(self, inputSize: int, clsNum: int, hiddenSize: int = 64) -> None:
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size = inputSize,
            hidden_size = hiddenSize,
            batch_first = True,
        )
        self.linear = nn.Linear(hiddenSize, clsNum)

    def forward(self, inputs: torch.Tensor):
        '''
        Args:
            inputs: 输入，形状`[batchSize, seqLen, inputSize]`

        Returns: 
            分类结果，形状`[batchSize, clsNum]`
        '''

        lstmOutput, _ = self.LSTM(inputs)  # [batchSize, seqLen, hiddenSize]
        lstmOutputMean = lstmOutput.mean(dim = 1)  # [batchSize, hiddenSize]
        linearOutput = self.linear(lstmOutputMean)  # [batchSize, clsNum]
        return linearOutput
    
    def predict(self, inputs):
        output: torch.Tensor = self(inputs)
        return output.argmax(dim = -1).to(device = 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
def bestNHead(inputSize: int):
    return min((
        i for i in range(1, inputSize + 1) if (inputSize // i) * i == inputSize
    ), key = lambda i: abs((i * i - inputSize) / (i * i)))

class TransAm(nn.Module):
    '''
    基于attention机制的序列分类模型。
    '''

    def __init__(self, inputSize: int, clsNum: int, num_layers: int = 2, dropout: float = 0.02):
        super().__init__()
        nHead = bestNHead(inputSize)
        print(f'n head = {nHead}')
        self.pos_encoder = PositionalEncoding(inputSize)  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = inputSize, nhead = nHead, dropout = dropout)  # nhead需要是inputSize的约数
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        self.decoder = nn.Linear(inputSize, clsNum)
        self.init_weights()

    def init_weights(self):  
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, 
        src: torch.Tensor, src_mask: torch.Tensor | None = None, 
        src_padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_padding)  
        decoderOutput = self.decoder(output.mean(dim = 1))
        return decoderOutput
    
    def predict(self, inputs: torch.Tensor):
        output: torch.Tensor = self(inputs)
        # print(f'output = {output}')
        # print(f'output shape = {output.shape}')
        # import time
        # time.sleep(3)
        return output.argmax(dim = -1).to(device = 'cpu')
    