import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print("n_parameter:", count_parameters(_model))
            _loss = None
            if not args.test_only:
                if args.auxiliary_out:
                    _loss = loss.ManualLossWithAuxiliary(args, checkpoint)
                else:
                    _loss = loss.Loss(args, checkpoint)
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
