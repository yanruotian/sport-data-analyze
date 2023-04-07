import torch

from torch import nn


class SeqCls(nn.Module):

    def __init__(self, inputSize: int, clsNum: int, hiddenSize: int = 64) -> None:
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size = inputSize,
            hidden_size = hiddenSize,
            batch_first = True,
        )
        self.linear = nn.Linear(hiddenSize, clsNum)

    def forward(self, inputs):
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
    