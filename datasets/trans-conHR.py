import csv

from tqdm import tqdm

INPUT_FILE = '连续心率数据.csv'
OUTPUT_FILE = '连续心率数据-transed.csv'

DATA_HEAD = '数据'

def batchYielder(inputList: list, batchSize: int):
    itemList = []
    for item in inputList:
        itemList.append(item)
        if len(itemList) >= batchSize:
            yield itemList
            itemList = []
    if len(itemList) > 0:
        yield itemList

reader = csv.DictReader(open(
    INPUT_FILE, 'r', encoding = 'utf-8-sig'
), delimiter = ',')

writerHeads = [head for head in reader.fieldnames if head != DATA_HEAD] + ['num-1', 'num-2', 'num-3']
writer = csv.DictWriter(open(
    OUTPUT_FILE, 'w', encoding = 'utf-8', newline = ''
), writerHeads, delimiter = ',')

writer.writeheader()

for line in reader:
    data = line[DATA_HEAD]
    dataList = list(eval(data))
    assert(len(dataList) % 3 == 0)
    for batchedData in tqdm(batchYielder(dataList, 3)):
        num1, num2, num3 = batchedData
        newLine = {
            key: value for key, value in line.items() if key != DATA_HEAD
        }
        newLine['num-1'] = int(str(num1).replace('-', ''), base = 16)
        newLine['num-2'] = int(str(num2).replace('-', ''), base = 16)
        newLine['num-3'] = int(str(num3).replace('-', ''), base = 16)
        writer.writerow(newLine)
