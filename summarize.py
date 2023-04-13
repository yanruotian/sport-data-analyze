'''
汇总输出文件夹的对比信息。
'''

import os
import csv
import json

from typing import Any, Dict, List
from tabulate import tabulate
from functools import reduce

SORT_HEAD = 'macro-f1'
INPUT_DIR = '/data/private/yanruotian/sport-data-analyze/outputs/conHR-fake/lstm'
RESULT_JSON = 'test-result.json'

resultDicts: List[Dict[str, Any]] = []

def getHeads() -> List[str]:
    keys = reduce(
        lambda setA, setB: setA | setB,
        (dict.keys() for dict in resultDicts),
        set(),
    )
    anyHeads = []
    floatHeads = []
    for key in keys:
        if reduce(
            lambda boolA, boolB: boolA and boolB,
            (isinstance(dict.get(key, None), float) for dict in resultDicts),
            True,
        ):
            floatHeads.append(key)
        else:
            anyHeads.append(key)
    return anyHeads + floatHeads

def joinDicts(dictA: dict, dictB: dict):
    return {
        key: dictA.get(key, None) or dictB.get(key, None) for key in (
            dictA.keys() | dictB.keys()
        )
    }

def summarize(path: str, depth: int = 0, preDict: Dict[str, Any] = dict()):
    if os.path.isdir(path):
        possibleJsonPath = os.path.join(path, RESULT_JSON)
        if os.path.isfile(possibleJsonPath):
            with open(possibleJsonPath, 'r') as file:
                resultObj = json.load(file)
            resultDicts.append(joinDicts(preDict, {
                'accuracy': float(resultObj.get('accuracy')),
                'macro-f1': float(resultObj.get('macro avg').get('f1-score')),
                'micro-f1': float(resultObj.get('weighted avg').get('f1-score')),
            }))
        else:
            for dirName in os.listdir(path):
                if '=' in dirName:
                    key, value = dirName.split('=')
                else:
                    key = f'null-{depth}'
                    value = dirName
                summarize(
                    os.path.join(path, dirName), 
                    depth = depth + 1, 
                    preDict = joinDicts(preDict, {key: value}),
                )

def main():
    summarize(INPUT_DIR)
    heads = getHeads()
    resultDicts.sort(key = lambda item: (
        -float(item.get(SORT_HEAD)) if SORT_HEAD in item else float('inf')
    ))
    with open(os.path.join(INPUT_DIR, 'summary.csv'), 'w') as file:
        writer = csv.DictWriter(file, heads, delimiter = ',')
        writer.writeheader()
        writer.writerows(resultDicts)
    table = tabulate(
        (tuple(dict.get(head, None) for head in heads) for dict in resultDicts),
        headers = heads, tablefmt = 'grid'
    )
    print(table)

if __name__ == '__main__':
    main()
    