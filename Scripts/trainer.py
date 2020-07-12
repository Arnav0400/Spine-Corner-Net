import os
import torch
import importlib
import torch.nn as nn
from dataloader import provider
import logging
import time
from tqdm.notebook import tqdm
import sys
sys.path.insert(0,'../models/')
from py_utils.kp_utils import _decode, _decode_val

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()
        self.model = model
        self.loss = loss
    def forward(self, xs, ys = None, **kwargs):
        preds = self.model(*xs, **kwargs)
        if ys != None:
            loss  = self.loss(preds, ys, **kwargs)
            return loss, preds
        return preds

class Average_Meter():
    def __init__(self):
        self.val = 0
        self.n = 0
    def reset(self):
        self.val = 0
        self.n = 0
    def update(self, val, n=1):
        self.val += val*n
        self.n+=n
    def avg(self):
        return self.val/self.n

class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, optim, loss, init_lr, bs, name):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = loss
        self.network = Network(self.model, self.criterion)
        self.criterion = loss
        self.optimizer = optim(self.model.parameters(), init_lr)
        self.phases = ["train", "val", "test"]
        self.device = torch.device("cuda:0")
        self.num_epochs = 0
        self.best_loss = 100.
        self.name = name
        self.batch_size = bs
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.network = self.network.to(self.device)
        torch.backends.cudnn.benchmark = True
        
        self.dataloaders = {
            phase: provider(
                phase=phase,
                batch_size=bs,
                num_workers=0,
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
#         smape_meter =  Average_Meter()
        print(f"Starting epoch: {epoch} | phase: {phase}")
        batch_size = self.batch_size
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            xs, ys = batch['xs'], batch['ys']
            xs = [x.cuda() for x in xs]
            ys = [y.cuda() for y in ys]
#             xs = xs.to(self.device)
#             ys = yx.to(self.device)
            
            if phase == "train":
                    loss, preds = self.network(xs, ys)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                preds = self.network(xs)
                preds = _decode_val(*preds)
                loss = self.criterion(preds, ys)
            
            loss_meter.update(loss.mean().item(),len(loss))
#             xs = xs.detach().cpu()
#             ys = ys.detach().cpu()
#             preds = preds.detach().cpu()
#             smape = self.prepare_for_smape(preds,ys, xs)
#             smape_meter.update(smape,ys.shape[0])
            tk0.set_postfix(loss=loss_meter.avg())
        return loss_meter.avg()
    
    def test(self, phase='test'):
        mse_meter =  Average_Meter()
        batch_size = self.batch_size
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            xs, ys = batch['xs'], batch['ys']
            xs = [x.cuda() for x in xs]
            ys = [y.cuda() for y in ys]
            preds = self.network([xs[0]])
            preds_scores, preds_points, preds_heats = _decode(*preds[-12:])
            xs = [x.detach().cpu() for x in xs]
            ys = [y.detach().cpu() for y in ys]
            preds_scores = [pred.detach().cpu() for pred in preds_scores]
            preds_points = [pred.detach().cpu() for pred in preds_points]
            preds_heats = [pred.detach().cpu() for pred in preds_heats]
#             smape = self.prepare_for_smape(preds,ys, xs)
#             smape_meter.update(smape,ys.shape[0])
        return preds_scores, preds_points, preds_heats
             
    def fit(self, epochs):
        self.num_epochs+=epochs
        for epoch in range(self.num_epochs-epochs, self.num_epochs):
            self.network.train()
            train_loss = self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.network.eval()
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
#             val_loss = train_loss
            if val_loss <= self.best_loss:
                print("* New optimal found according, saving state *")
                state["best_loss"] = self.best_loss = val_loss
                os.makedirs('models/', exist_ok=True)
                torch.save(state, 'models/'+self.name+'.pth')
            content =  time.ctime() + ' ' + f'Epoch {epoch}, lr: {self.optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, val loss: {val_loss:.5f}'
            print(content)
            os.makedirs('logs/', exist_ok=True)
            with open(f'logs/log_{self.name}.txt', 'a') as appender:
                appender.write(content + '\n')

