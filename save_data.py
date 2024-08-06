import os
import argparse
import FinanceDataReader as fdr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./data/tmp')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    Arguments
    '''
    args = parse_args()
    arg_str = 'Arguments:\n'
    for arg in vars(args):
        arg_str += '%20s: %s\n' % (arg, getattr(args, arg))
    print(arg_str)

    '''
    Save Stock List
    '''
    print('saving stock list... ', flush=True, end='')
    os.makedirs(args.out_dir, exist_ok=True)
    stocks_path = os.path.join(args.out_dir, 'stock_list.csv')
    stocks = fdr.StockListing('KRX')
    stocks.to_csv(stocks_path)
    print('done!', flush=True)

    '''
    Save Each Stock Data
    '''
    print('saving each stock data...', flush=True)
    for i, code in enumerate(stocks['Code']):
        if i % 100 == 0:
            print('  %d / %d' % (i, len(stocks)))

        df = fdr.DataReader(code)
        out_path = os.path.join(args.out_dir, code + '.csv')
        df.to_csv(out_path)
    print('Finish!')




