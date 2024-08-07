import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader

from model import TFModel
from stock_dataset import StockDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # directories
    parser.add_argument('--data_dir', type=str, default='./data/v1/20240725')
    parser.add_argument('--save_dir', type=str, default='./weights/recent/')

    # dataset
    parser.add_argument('--inp_dim', type=int, default=60)
    parser.add_argument('--out_dim', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    # optimizer
    parser.add_argument('--num_iter', type=int, default=100_000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.02, help='SmoothL1Loss')
    parser.add_argument('--step_size', type=int, default=10_000, help='stepLR')
    parser.add_argument('--gamma', type=float, default=0.9, help='stepLR')
    parser.add_argument('--clip_grad_norm', type=float, default=1.)

    # training
    parser.add_argument('--print_interval', type=int, default=100, help='unit: iter')
    parser.add_argument('--save_interval', type=int, default=10_000, help='unit: iter')

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
    Dataset
    '''
    dataset = StockDataset(
        args.data_dir,
        args.inp_dim,
        args.out_dim,
        # code_list=['005930'],  # samsung
        code_list=['373220'],  # lgensol
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True)
    print('len(dataset):', len(dataset))
    print('len(dataloader):', len(dataloader))

    '''
    Model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFModel(args.inp_dim, args.out_dim).to(device)
    model.train()

    '''
    Optimizer
    '''
    criterion = torch.nn.SmoothL1Loss(beta=args.beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    '''
    Save Directory
    '''
    os.makedirs(args.save_dir, exist_ok=True)

    '''
    Training
    '''
    i = 0
    is_done = False
    while True:
        for inp, tgt in dataloader:
            # forward and update
            out = model(inp.to(device))
            loss = criterion(out, tgt.to(device))
            assert not torch.isinf(loss)
            assert not torch.isnan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # print
            if i % args.print_interval == 0:
                log_str = datetime.datetime.now().strftime('[%H:%M:%S]')
                log_str += ' iter: %d, loss: %.6f' % (i, loss)
                log_str += ', lr: %.6f' % scheduler.get_last_lr()[0]
                print(log_str, flush=True)

            # save weights
            if (i + 1) % args.save_interval == 0:
                weight_path = os.path.join(args.save_dir, '%08d.pth' % (i + 1))
                torch.save(model.state_dict(), weight_path)
                log_str = '======== %s saved. ========' % weight_path
                print(log_str, flush=True)

            # lr scheduler
            scheduler.step()

            # update iteration
            i += 1
            if i == args.num_iter:
                is_done = True
                break

        if is_done:
            break

    print('Finish!')


