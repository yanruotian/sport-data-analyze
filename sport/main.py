import os
import json
import torch
import readline

from torch import nn
from typing import List, Type, Callable
from collections import defaultdict

from random import seed as randomSeed

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split

from numpy.random import seed as numpySeed

from .plot import draw
from .model import SeqCls, TransAm
from .dataset import SportDataset, SEQ_HEADS

def setSeed(seed: int):
    randomSeed(seed)
    numpySeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def summarizeList(list: List[int]):
    resultDict = defaultdict(int)
    for item in list:
        resultDict[item] += 1
    return dict(resultDict)

def splitDataset(dataset: SportDataset, args):
    trainSize = int(args.train_proportion * len(dataset))
    evalTestSize = len(dataset) - trainSize
    evalSize = int((args.eval_test_rate / (args.eval_test_rate + 1)) * evalTestSize)
    testSize = evalTestSize - evalSize
    return random_split(
        dataset, (trainSize, evalSize, testSize),
        generator = torch.Generator().manual_seed(args.seed),
    )

def train(
    model: SeqCls, 
    dataset: SportDataset, 
    lossFunc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    optimizer, args,
    criteriaFunc: Callable[[dict], float] = (lambda dict: dict.get('accuracy')),
    criteriaValueList: List[float] = [],
    lossList: List[float] = [],
):
    trainData, evalData, testData = splitDataset(dataset, args)
    stepCount = 0
    earlyStopCount, lastValue = 0, None
    bestValue, bestModelPath = None, None
    for epoch in range(args.epoch):
        model.train()
        for step, batch in enumerate(DataLoader(trainData, args.batch_size, shuffle = True)):
            optimizer.zero_grad()
            tensorBatch, labelBatch = batch
            output = model(tensorBatch)
            loss = lossFunc(output, labelBatch.to(device = args.device))
            lossList.append(loss.item())
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch :2d} | step {step :3d} | loss {loss}')
            stepCount += 1
            if stepCount % args.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    true, pred = [], []
                    for batch in DataLoader(evalData, args.batch_size):
                        tensorBatch, labelBatch = batch
                        true.extend(labelBatch.tolist())
                        pred.extend(model.predict(tensorBatch).tolist())
                savePath = os.path.join(args.output_path, 'models', f'{stepCount :04d}.pt')
                os.makedirs(os.path.dirname(savePath), exist_ok = True)
                torch.save(model.state_dict(), savePath)
                dictResult = classification_report(true, pred, output_dict = True)
                value = criteriaFunc(dictResult)
                if lastValue is not None and value <= lastValue:
                    earlyStopCount += 1
                else:
                    earlyStopCount = 0
                lastValue = value
                criteriaValueList.append(value)
                if bestValue is None or value > bestValue:
                    bestValue = value
                    bestModelPath = savePath
                print('----- Eval -----')
                print(f'true: {summarizeList(true)}')
                print(f'pred: {summarizeList(pred)}')
                print(classification_report(true, pred))
                model.train()
        if earlyStopCount >= args.early_stopping:
            print(f'early stop at epoch {epoch} (early stopping = {args.early_stopping}, stopping count = {earlyStopCount})')
            break
    if bestModelPath is not None:
        print(f'loading best model from "{bestModelPath}"')
        model.load_state_dict(torch.load(bestModelPath))
        model.eval()
        with torch.no_grad():
            true, pred = [], []
            for batch in DataLoader(testData, args.batch_size):
                tensorBatch, labelBatch = batch
                true.extend(labelBatch.tolist())
                pred.extend(model.predict(tensorBatch).tolist())
        print('----- Test -----')
        print(f'true: {summarizeList(true)}')
        print(f'pred: {summarizeList(pred)}')
        print(classification_report(true, pred))
        with open(os.path.join(args.output_path, 'test-result.json'), 'w') as file:
            json.dump(classification_report(
                true, pred, output_dict = True
            ), file, ensure_ascii = False)

def mainTrain(args):
    setSeed(args.seed)
    os.makedirs(args.output_path, exist_ok = True)
    device = torch.device(args.device)
    dataset = SportDataset(args.input_path, args.seq_len, device)
    modelType: Type[nn.Module] = {
        'lstm': SeqCls,
        'transformer': TransAm,
    }.get(args.model_type)
    model = modelType(
        inputSize = len(SEQ_HEADS),
        clsNum = len(dataset.clsTypes),
    ).to(device)
    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = args.lr, betas = (0.9, 0.999)
    )
    criteriaValues, lossValues = [], []
    train(
        model = model,
        dataset = dataset,
        lossFunc = lossFunc,
        optimizer = optimizer,
        args = args,
        criteriaFunc = {
            'accuracy': (lambda dict: dict.get('accuracy')),
            'macro-f1': (lambda dict: dict.get('macro avg').get('f1-score')),
            'micro-f1': (lambda dict: dict.get('weighted avg').get('f1-score')),
        }.get(args.criteria),
        criteriaValueList = criteriaValues,
        lossList = lossValues,
    )
    draw(
        criteriaValues, 
        os.path.join(args.output_path, 'pngs', f'{args.criteria}.png'),
    )
    draw(
        lossValues, 
        os.path.join(args.output_path, 'pngs', f'loss.png'),
    )

def main(args):
    mode: str = args.mode
    if mode == 'train':
        mainTrain(args)
    elif mode == 'i-predict':
        mainIPredict(args)
    else:
        print(f'unsupported mode: {mode}')


PREDICT_DICT = {
    'lfw0b43pkwp0': 0, 'z40j59eifo85': 1, 'wbzacaofn94g': 2, 'si0y4kl9omvy': 3, 'inr26rmgidwx': 4, 'htvpwpb5aafl': 5, 'k38asku1lu74': 6, '8i91ltam916f': 7, 'fnuba4pgq27v': 8, 'nkkvdlu8dou7': 9, '1wc8auern8m3': 10, 'szy72isrfejp': 11, 
    'vz3601s7g83t': 12, 'i4tp4aoltxmt': 13, 'cy3l8w6ha539': 14, 'p5j807r8b4va': 15, '7c8q60ulbk6z': 16, 'fu8er5d20z24': 17, 'y0zhwt78b52z': 18, '44sq6x2mkqbm': 19, 'qb00avvyx7qg': 20, 'ghewogphloe6': 21, 'dj21vpnzb64s': 22, 'j0csc3p7qnv6': 23, 'by7wui29w0k5': 24, 'ss1xqchxxgp8': 25, 
    'yd1bzm0eqsgk': 26, 'w8yyb81gssry': 27, '1kf2s9uqajcc': 28, 'nk8tmb2we98q': 29, '48amhzr91v6y': 30, 'afxt0mo9w6b3': 31, '9g3auj21gc0g': 32, 'y6d0px90vwfe': 33, 'fk4ajpmie2mf': 34, 'aeuksddgoqti': 35, 'q4jtplrsog9e': 36,
    'f0sy5trjkc5d': 37, '4na6mae3nrqd': 38, '0v4x1y2orxtw': 39, 'fjn044ww10bk': 40, 'w43ey7lsn17e': 41, '1gdhrlimd8uz': 42, 'vxg5nxskf5ur': 43, 'zhhmhwl4aqnb': 44, 'ixdgra4112vo': 45, 'vsecn8voe3fr': 46, 'e9r3ja41q289': 47, 'gixt35u1izh7': 48, 
    'afbmdy1fwfp5': 49, '7mull4siaunn': 50, '17td3419b3he': 51, '343jo1ql3smt': 52, 'slmwb5s0k1nb': 53, 'k8fvmppzljsc': 54, 'lstsrlxkvl1y': 55, '677ba49wir1t': 56, '967gqiujnp0p': 57, '9wkpsvmmwis0': 58, 'qwnp27iv5o5g': 59, 'i6jryjc0nw3e': 60, 
    'ee031wizw8gk': 61, 'n9tcspumphp0': 62, 'z8vgs3wcwzl1': 63, 'mmnlbw6yxyrf': 64, 'oa1rvw0nde9s': 65, 'bk5ng7qyk9q9': 66, 'ru3pomjgeu14': 67, '00rqww8xgysa': 68, 'hn9qkh6i3kic': 69, 'pd9q9l8saw0v': 70, 'g1wmqimoghbs': 71, 'sui5rciafuo5': 72, '70s7tx7k22ig': 73, '4mv543gtysjt': 74,
}

def readInputLine(line: str, inputs: List[List[int]]):
    try:
        inputs.append(list(map(int, line.split(','))))
    except Exception as e:
        print(f'err occur in line "{line}"')
        print(f'exception: {e}')

def mainIPredict(args):
    setSeed(args.seed)
    device = torch.device(args.device)
    clsNumDict = {
        value: key for key, value in PREDICT_DICT.items()
    }
    modelType: Type[nn.Module] = {
        'lstm': SeqCls,
        'transformer': TransAm,
    }.get(args.model_type)
    model = modelType(
        inputSize = 3,
        clsNum = len(clsNumDict),
    ).to(device)
    model.load_state_dict(torch.load(args.input_path))
    inputs: List[int] = []
    model.eval()
    with torch.no_grad():
        while True:
            line = input('> ').strip()
            if line == '.end':
                modelOutput: torch.Tensor = model(
                    torch.as_tensor([inputs], device = device).float()
                )
                clsResult = modelOutput.argmax(dim = -1).tolist()[0]
                print(f'cls result: {clsNumDict.get(clsResult)}')
                inputs = []
            elif line == '.exit':
                print('exiting...')
                break
            elif line.startswith('.read '):
                filePath = line[len('.read ') : ].strip()
                with open(filePath, 'r') as file:
                    for line in file:
                        readInputLine(line, inputs)
            else:
                readInputLine(line, inputs)
