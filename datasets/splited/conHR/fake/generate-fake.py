import os
import csv
import random

from tqdm import tqdm

import numpy as np

HEADS = [
    '名称', '手环', '采集时间', '开始时间', '结束时间', 
    '入库时间', 'num-1', 'num-2', 'num-3',
]

ID_HEAD = '手环'

FILE_NUM = 75
DATA_PER_FILE = 5000

RANDOM_CHARS = list({
    chr(ord('0') + i) for i in range(10)
} | {
    chr(ord('a') + i) for i in range(26)
})

def randomString(length: int = 12):
    return ''.join(random.choice(RANDOM_CHARS) for _ in range(length))

def generateOne(fileId: int, idString: str | None = None):
    if idString is None:
        idString = randomString()
    with open(f'{idString}.csv', 'w', newline = '', encoding = 'utf-8') as file:
        writer = csv.DictWriter(file, HEADS, delimiter = ',')
        writer.writeheader()
        for _ in range(DATA_PER_FILE):
            line = {
                head: 'null' for head in HEADS
            }
            line[ID_HEAD] = idString
            line['num-1'] = int(np.random.normal(loc = 40 + fileId * 200 / FILE_NUM, scale = 4))
            line['num-2'] = int(np.random.normal(loc = 60 + fileId * 80 / FILE_NUM, scale = 1))
            line['num-3'] = 1 if np.random.uniform(0, 1) < 4 / 7000 else 0
            writer.writerow(line)

def main():
    fileDir = os.path.dirname(os.path.abspath(__file__))
    for fileName in os.listdir(fileDir):
        filePath = os.path.join(fileDir, fileName)
        if os.path.isfile(filePath) and fileName.endswith('.csv'):
            os.remove(filePath)
    for i in tqdm(range(FILE_NUM)):
        generateOne(i)

if __name__ == '__main__':
    main()
