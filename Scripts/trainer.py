import os
import torch
import importlib
import torch.nn as nn
from dataloader import provider
import logging
import time
import sys
sys.path.insert(0,'../models')
from models.py_utils.kp_utils import _decode

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss  = self.loss(preds, ys, **kwargs)
        return loss, preds

class Average_Meter():
    def __init__():
        self.val = 0
        self.n = 0
    def reset():
        self.val = 0
        self.n = 0
    def update(val, n=1):
        self.val += val*n
        self.n+=n
    def avg():
        return self.val/self.n

class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, optim, loss, lr, bs, name):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = loss
        self.network = Network(self.model, self.loss)
        self.optimizer = optim
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        self.num_epochs = 0
        self.best_smape = 0.
        self.name = name
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.network = self.network.to(self.device)
        cudnn.benchmark = True

        self.dataloaders = {
            phase: provider(
                phase=phase,

                crop_type=crop_type,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers if phase=='train' else 0,
            )
            for phase in self.phases
        }

    def load_model(self, name, path='models/'):
        state = torch.load(path+name, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print("Loaded model")

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def cal_smape(detections, detections_gt):
        [tl_xs, tl_ys, br_xs, br_ys, tr_xs, tr_ys, bl_xs, bl_ys] = detections
        [tl_xs_gt, tl_ys_gt, br_xs_gt, br_ys_gt, tr_xs_gt, tr_ys_gt, bl_xs_gt, bl_ys_gt] = detections_gt
        

    def prepare_for_smape(self, preds, ys, xs):
        _, detections = _decode(*preds[-12:])
        [tl_heatmaps, br_heatmaps, tr_heatmaps, bl_heatmaps, tag_mask, tl_regr, br_regr, tr_regr, bl_regr] = ys
        [_, tl_tag, br_tag, tr_tag, bl_tag] = xs
        _, detections_gt = _decode(tl_heatmaps, br_heatmaps, tr_heatmaps, bl_heatmaps, tl_tag, br_tag, tr_tag, bl_tag, tl_regr, br_regr, tr_regr, bl_regr)
        return self.cal_smape(detections, detections_gt)

    def iterate(self, epoch, phase):
        loss_meter = Average_Meter()
        smape_meter =  Average_Meter()
        print(f"Starting epoch: {epoch} | phase: {phase}")
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            xs, ys = batch
            xs = xs.to(self.device)
            ys = yx.to(self.device)
            loss, preds = self.network(xs, ys)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_meter.update(loss.mean().item(),len(loss))
            xs = xs.detach().cpu()
            ys = ys.detach().cpu()
            preds = preds.detach().cpu()
            smape = self.prepare_for_smape(preds,ys, xs)
            smape_meter.update(smape,ys.shape[0])
            tk0.set_postfix(loss=loss_meter.avg(), smape = smape_meter.avg())
        return loss_meter.avg(), smape_meter.avg()

    def fit(self, epochs):
        self.num_epochs+=epochs
        for epoch in range(self.num_epochs-epochs, self.num_epochs):
            self.net.train()
            train_loss, train_smape = self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_smape": self.best_smape,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.net.eval()
            with torch.no_grad():
                val_loss, val_smape = self.iterate(epoch, "val")
            if val_smape > self.best_smape:
                print("* New optimal found according, saving state *")
                state["best_smape"] = self.best_smape = val_smape
                os.makedirs('models/', exist_ok=True)
                torch.save(state, 'models/'+self.name+'.pth')
            content =  time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, val loss: {val_loss:.5f}, val_smape: {(val_smape):.5f}'
            print(content)
            os.makedirs('logs/', exist_ok=True)
            with open(f'logs/log_{self.name}.txt', 'a') as appender:
                appender.write(content + '\n')

