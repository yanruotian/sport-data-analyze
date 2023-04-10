import os
import csv
import torch

from typing import Dict, List, DefaultDict
from collections import defaultdict

from torch.utils.data import Dataset

from .args import ARGS

def getSeqHeads():
    datasetType: str = ARGS.dataset
    if datasetType == 'comprehensive':
        return [
            '温度', 'hrv', 'cvrr',
            '低压', '高压', '血氧',
        ]
    elif datasetType == 'conHR':
        from .configs.conHR import SEQ_HEADS
        return SEQ_HEADS
    else:
        print(f'unrecognized dataset: {datasetType}')
        assert(False)

LABEL_STR_HEAD = '手环'
LABEL_HEAD = 'label'
# SEQ_HEADS = [
#     '步数', '卡路里', '温度', '心率', 'hrv', 
#     '低压', '高压', '血氧', 'cvrr', '体质因子',
# ]
SEQ_HEADS = getSeqHeads()

def readCsv(path: str, topDir: bool = True):
    if os.path.isfile(path) and path.endswith('.csv'):
        with open(path, 'r') as file:
            reader = csv.DictReader(file, delimiter = ',')
            for line in reader:
                yield line
    elif os.path.isdir(path) and (topDir or os.path.basename(path) != 'fake'):
        for fileName in os.listdir(path):
            for line in readCsv(os.path.join(path, fileName), False):
                yield line


class SportDataset(Dataset):

    def __init__(self, inputPath: str, seqLen: int, device = None) -> None:
        super().__init__()
        rawData = list(readCsv(inputPath))
        self.clsTypes = list(set(map(lambda line: line.get(LABEL_STR_HEAD), rawData)))
        self.clsDict = {type: i for i, type in enumerate(self.clsTypes)}
        self.data = [
            (
                self.toTensor(data, device), 
                int(data[0].get(LABEL_HEAD)),
            ) for data in self.dealData(
                rawData, seqLen, self.clsDict
            )
        ]
        print(f'dataset ready! len = {len(self)}, clsDict = {self.clsDict}')

    def __getitem__(self, index: int):
        tensors, label = self.data[index]
        return tensors, int(label)
    
    def __len__(self):
        return len(self.data)
    
    def summarize(self):
        summarizeDict = defaultdict(int)
        for _, label in self.data:
            summarizeDict[label] += 1
        return dict(summarizeDict)

    @classmethod
    def toTensor(cls, data: List[Dict[str, str]], device):
        return torch.as_tensor(
            list(map(lambda line: (
                [float(line.get(head)) for head in SEQ_HEADS]
            ), data)),
            device = device,
        ).float()

    @classmethod
    def dealData(cls, data: List[Dict[str, str]], seqLen: int, labelDict: Dict[str, int], strict: bool = True):
        resultDict: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
        for line in data:
            labelStr = line.get(LABEL_STR_HEAD)
            line[LABEL_HEAD] = labelDict.get(labelStr)
            resultDict[labelStr].append(line)
            if len(resultDict[labelStr]) >= seqLen:
                yield resultDict[labelStr]
                resultDict[labelStr] = []
        if not strict:
            for result in resultDict.values():
                if len(result) > 0:
                    yield result
    