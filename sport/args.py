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
    parser.add_argument(
        '--dataset', type = str, default = 'comprehensive', help = 'dataset type',
        choices = ('comprehensive', 'conHR')
    )
    parser.add_argument(
        '--model-type', type = str, default = 'lstm', help = 'decide which type of model to use',
        choices = ('lstm', 'transformer')
    )
    parser.add_argument(
        '--criteria', type = str, default = 'accuracy', help = 'criteria to judge the best model from eval results',
        choices = ('accuracy', 'macro-f1', 'micro-f1')
    )
    parser.add_argument('--early-stopping', type = float, default = float('inf'), help = 'tolerance of eval criteria decrease')
    return parser.parse_args()

ARGS = getArgs()
