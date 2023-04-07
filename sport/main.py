import os
import torch

from random import seed as randomSeed

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split

from numpy.random import seed as numpySeed

from .model import SeqCls
from .dataset import SportDataset, SEQ_HEADS

def setSeed(seed: int):
    randomSeed(seed)
    numpySeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def splitDataset(dataset: SportDataset, args):
    trainSize = int(args.train_proportion * len(dataset))
    evalTestSize = len(dataset) - trainSize
    evalSize = int((args.eval_test_rate / (args.eval_test_rate + 1)) * evalTestSize)
    testSize = evalTestSize - evalSize
    return random_split(
        dataset, (trainSize, evalSize, testSize),
        generator = torch.Generator().manual_seed(args.seed),
    )

def train(model: SeqCls, dataset: SportDataset, lossFunc, optimizer, args):
    trainData, evalData, testData = splitDataset(dataset, args)
    stepCount = 0
    bestValue, bestModelPath = None, None
    for epoch in range(args.epoch):
        model.train()
        for step, batch in enumerate(DataLoader(trainData, args.batch_size, shuffle = True)):
            optimizer.zero_grad()
            tensorBatch, labelBatch = batch
            output = model(tensorBatch)
            loss = lossFunc(output, labelBatch.to(device = device))
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
                savePath = os.path.join(args.output_path, f'{stepCount :04d}.pt')
                torch.save(model.state_dict(), savePath)
                dictResult = classification_report(true, pred, output_dict = True)
                value = dictResult.get('accuracy')
                if bestValue is None or value > bestValue:
                    bestValue = value
                    bestModelPath = savePath
                print('----- Eval -----')
                print(classification_report(true, pred))
                model.train()
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
            print(classification_report(true, pred))

def main(args):
    setSeed(args.seed)
    os.makedirs(args.output_path, exist_ok = True)
    device = torch.device(args.device)
    dataset = SportDataset(args.input_path, args.seq_len, device)
    model = SeqCls(
        inputSize = len(SEQ_HEADS),
        clsNum = len(dataset.clsTypes),
    ).to(device)
    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = args.lr, betas = (0.9, 0.99)
    )
    train(
        model = model,
        dataset = dataset,
        lossFunc = lossFunc,
        optimizer = optimizer,
        args = args
    )
