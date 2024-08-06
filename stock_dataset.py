import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data_dir, inp_dim, out_dim, code_list=None,
                 features=['Open', 'High', 'Low', 'Close', 'Volume']):
        super().__init__()

        # parameters
        self.data_dir = data_dir
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.features = features

        # code list
        if code_list is None:
            stock_list_path = os.path.join(data_dir, 'stock_list.csv')
            df = pd.read_csv(stock_list_path)
            self.code_list = sorted(df['Code'].values.tolist())
        else:
            self.code_list = code_list

        # check data size and keep cumsum
        sizes = []
        for code in self.code_list:
            stock_path = os.path.join(data_dir, code + '.csv')
            df = pd.read_csv(stock_path)
            size = max(len(df) - inp_dim - out_dim + 1, 0)
            sizes.append(size)
        self.cumsum_size = np.cumsum(sizes)

    def __len__(self):
        return self.cumsum_size[-1]

    def __getitem__(self, index):
        assert index < self.cumsum_size[-1]

        # indicies
        code_idx = np.count_nonzero(self.cumsum_size <= index)
        ts = index if code_idx == 0 else index - self.cumsum_size[code_idx-1]
        ti = ts + self.inp_dim
        to = ti + self.out_dim

        # read data
        stock_path = os.path.join(self.data_dir, self.code_list[code_idx] + '.csv')
        df = pd.read_csv(stock_path)
        inp = df[ts:ti][self.features].values.astype(np.float32)
        out = df[ti:to]['Close'].values.astype(np.float32)
        assert len(inp) == self.inp_dim
        assert len(out) == self.out_dim

        # normaliation
        price_mean = inp[:, 3].mean()
        volumn_mean = inp[:, 4].mean()
        assert not np.isclose(price_mean, 0.)
        inp[:, :4] /= price_mean
        out /= price_mean
        if not np.isclose(volumn_mean, 0.):
            inp[:, 4] /= volumn_mean
        
        return inp, out


if __name__ == '__main__':
    data_dir = './data/v1/20230103'
    dataset = StockDataset(data_dir, 100, 50)
    print('len(dataset):', len(dataset))


