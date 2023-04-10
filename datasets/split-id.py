import os
import csv
import json

from collections import defaultdict

INPUT_CSV = '连续心率数据-transed.csv'
OUTPUT_DIR = 'splited/连续心率数据'

os.makedirs(OUTPUT_DIR, exist_ok = True)

with open(INPUT_CSV, 'r', encoding = 'utf-8-sig') as file:
    reader = csv.DictReader(file, delimiter = ',')
    FIELD_NAMES = tuple(reader.fieldnames)
    DATA = list(reader)

WRITER_DICT = dict()
COUNT_DICT = defaultdict(int)

for line in DATA:
    id = line.get('手环')
    if id not in WRITER_DICT:
        writer = csv.DictWriter(
            open(
                os.path.join(OUTPUT_DIR, f'{id}.csv'), 
                'w+', newline = '', encoding = 'utf-8'
            ),
            FIELD_NAMES,
            delimiter = ',',
        )
        writer.writeheader()
        WRITER_DICT[id] = writer
    else:
        writer = WRITER_DICT.get(id)
    writer.writerow(line)
    COUNT_DICT[id] += 1

with open('count.json', 'w+', encoding = 'utf-8') as file:
    json.dump(COUNT_DICT, file, ensure_ascii = False)
    