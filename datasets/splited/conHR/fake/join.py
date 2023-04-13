import os
import csv

IGNORE_HEADS = {f'num-{i}' for i in (1, 2, 3)}
DATA_HEAD = '数据'
OUTPUT_FILE = 'fake'

writer = None

def writeOne(path: str):
    global writer
    with open(path, 'r') as file:
        reader = csv.DictReader(file, delimiter = ',')
        if writer is None:
            writer = csv.DictWriter(
                open(OUTPUT_FILE, 'w'), 
                list(set(reader.fieldnames) - IGNORE_HEADS) + [DATA_HEAD], 
                delimiter = ','
            )
            writer.writeheader()
        content = list(reader)
    dataList = []
    for line in content:
        try:
            dataList.extend(int(line.get(f'num-{i}')) for i in (1, 2, 3))
        except:
            print(path)
            print(line)
            exit()
    writerLine = {key: content[0].get(key, None) for key in writer.fieldnames}
    writerLine[DATA_HEAD] = str(dataList)
    writer.writerow(writerLine)

def main():
    for file in os.listdir('./'):
        if os.path.isfile(file) and file.endswith('.csv'):
            writeOne(file)

if __name__ == '__main__':
    main()
        