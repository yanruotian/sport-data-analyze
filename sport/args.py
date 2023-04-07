from argparse import ArgumentParser

def getArgs():
    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--eval-step', type = int, help = 'frequency of eval')
    parser.add_argument('--epoch', type = int, help = 'epoch count to train')
    parser.add_argument('--batch-size', type = int, help = 'batch size of training')
    parser.add_argument('--lr', type = float, help = 'leaning rate of training')
    parser.add_argument('-o', '--output-path', type = str, help = 'path to save models')
    parser.add_argument('-i', '--input-path', type = str, help = 'path to a csv file or a dir containing csv files')
    parser.add_argument('--seq-len', type = int, help = 'the seq len for split of data from the same person')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'use cpu or cuda')
    parser.add_argument('--train-proportion', type = float, help = 'the proportion of amount from whole data used for train')
    parser.add_argument('--eval-test-rate', type = float, default = 1., help = 'size(eval data) / size(test data)')
    return parser.parse_args()
