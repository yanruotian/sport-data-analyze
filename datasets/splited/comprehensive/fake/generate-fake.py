import os
import csv

import numpy as np

HEADS = [
    '学校', '班级', '学号', '姓名', '记录号', '记录批号', '学生ID', 
    '手机', '手环', '采集时间', '步数', '卡路里', '温度', '心率', 
    'hrv', '低压', '高压', '血氧', 'cvrr', '体质因子',
]

ID_HEAD = '手环'

FILE_NUM = 4
DATA_PER_FILE = 1000

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
    for i in range(FILE_NUM):
        generateOne(i)

if __name__ == '__main__':
    main()
