import os
import csv

from tqdm import tqdm

import numpy as np

HEADS = [
    '名称', '手环', '采集时间', '开始时间', '结束时间', 
    '入库时间', 'num-1', 'num-2', 'num-3',
]

ID_HEAD = '手环'

FILE_NUM = 75
DATA_PER_FILE = 5000

def generateOne(fileId: int, filePath: str | None = None):
    if filePath is None:
        filePath = f'{fileId :02d}.csv'
    with open(filePath, 'w', newline = '', encoding = 'utf-8') as file:
        writer = csv.DictWriter(file, HEADS, delimiter = ',')
        writer.writeheader()
        for _ in range(DATA_PER_FILE):
            line = {
                head: np.random.normal(loc = fileId) for head in HEADS
            }
            line[ID_HEAD] = fileId
            writer.writerow(line)

def main():
    for i in tqdm(range(FILE_NUM)):
        generateOne(i)

if __name__ == '__main__':
    main()
