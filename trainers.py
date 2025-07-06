import os
import collections

import numpy as np
import torch
from tqdm import tqdm
from metrics import cal_acc_precision_recall_f1
from sklearn.metrics import average_precision_score
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import math
import os
import logging
import json
import datetime
from accelerate import Accelerator

# 创建一个Formatter对象，包含时间信息
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 配置日志记录器并设置Formatter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PLMTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader,
                 lr=0.001, num_epochs=10, batch_size=128, save_path='./save/',
                 model_name='model', monitor='loss', average='binary',
                 criterion=None, device=None, fp16=False):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.save_path = save_path
        self.model = model
        # self.model.to(device)
        self.model_name = model_name
        self.device = device
        self.best_metric = -float('inf')
        self.monitor = monitor
        self.average = average
        self.model_save_path = os.path.join(save_path, model_name)
        self.fp16 = fp16

    def train(self, dataloader, optimizer, criterion=None, scheduler=None):
        train_loss = 0
        real_labels = []
        pred_labels = []
        pred_probs = []
        self.model.train()
        if not self.fp16:
            """全精度训练"""
            self.model.to(self.device)
            self.model.train()
            for batch in tqdm(dataloader):
                # 将梯度清零
                optimizer.zero_grad()
                # 将数据和标签移动到指定设备上
                batch = {k: batch[k].to(self.device) for k in batch}
                # 前向传播
                output = self.model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                train_loss += loss.item()
                _pred_probs = torch.softmax(output.logits, dim=-1)
                _pred_labels = torch.argmax(_pred_probs, dim=-1)
                real_labels += batch['labels'].cpu().numpy().tolist()
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.detach().cpu().numpy().tolist()
        else:
            """半精度训练"""
            for batch in tqdm(dataloader):
                batch.to(self.accelerator.device)
                # 将梯度清零
                optimizer.zero_grad()
                # 前向传播
                output = self.model(**batch)
                loss = output.loss
                self.accelerator.backward(loss)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                train_loss += loss.item()
                _pred_probs = torch.softmax(output.logits, dim=-1)
                _pred_labels = torch.argmax(_pred_probs, dim=-1)
                real_labels += batch['labels'].cpu().numpy().tolist()
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.detach().cpu().numpy().tolist()

        """指标计算"""
        metric = cal_acc_precision_recall_f1(real_labels, pred_labels, pred_probs=pred_probs, average=self.average)
        metric['loss'] = train_loss / len(dataloader)
        return metric

    def evaluate(self, dataloader, criterion=None, is_test=False):
        self.model.eval()
        valid_loss = 0
        real_labels = []
        pred_labels = []
        pred_probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # 将数据和标签移动到指定设备上
                batch = {k: batch[k].to(self.device) for k in batch}
                output = self.model(**batch)
                loss = output.loss
                valid_loss += loss.item()
                _pred_probs = torch.softmax(output.logits, dim=-1)
                _pred_labels = torch.argmax(_pred_probs, dim=-1)
                # _pred_labels = torch.argmax(output.logits, dim=-1)
                # _pred_probs = torch.softmax(output.logits, dim=-1)
                real_labels += batch['labels'].cpu().numpy().tolist()
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.cpu().numpy().tolist()
        # np.savez('./result/qianyi/yes.npz', real_labels=real_labels, pred_probs=pred_probs)
        metric = cal_acc_precision_recall_f1(real_labels, pred_labels, pred_probs=pred_probs, average=self.average, is_test=is_test)
        metric['loss'] = valid_loss / len(dataloader)
        return metric

    def evaluate_kfold(self, dataloader, criterion=None, is_test=False):
        self.model.eval()
        valid_loss = 0
        real_labels = []
        pred_labels = []
        pred_probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # 将数据和标签移动到指定设备上
                batch = {k: batch[k].to(self.device) for k in batch}
                output = self.model(**batch)
                loss = output.loss
                valid_loss += loss.item()
                _pred_labels = torch.argmax(output.logits, dim=-1)
                _pred_probs = torch.softmax(output.logits, dim=-1)[:,1]
                real_labels += batch['labels'].cpu().numpy().tolist()
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.cpu().numpy().tolist()
        return real_labels,pred_probs

    def save_best_model(self, save_path, curr_metric):
        if self.monitor == 'loss':
            curr_metric = -curr_metric
        if self.best_metric <= curr_metric:
            self.best_metric = curr_metric
            self.model.save_pretrained(save_path)

    def training(self):
        self.accelerator = Accelerator(mixed_precision="fp16")
        # self.accelerator = Accelerator(mixed_precision="fp16" if self.fp16 else "no")
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        update_parameters = []
        update_flag = False
        for n, p in self.model.named_parameters():
            if 'encoder.block.23.layer' in n:
                # if 'encoder.block.23.layer' in n:
                    update_flag = True
            if update_flag:
                p.requires_grad = True
                update_parameters.append((n, p))
            else:
                p.requires_grad = False
        # print(len(update_parameters))

        optimizer = AdamW([
            {'params': [param for name, param in update_parameters if 'encoder' in name], 'lr': self.lr},  # 预训练模型层
            {'params': [param for name, param in update_parameters if not 'encoder' in name], 'lr': self.lr * 5}  # 任务特定层
        ])


        # optimizer = AdamW([
        #     {'params': [param for name, param in update_parameters if 'encoder' in name], 'lr': self.lr, 'weight_decay': 1e-4},
        #     {'params': [param for name, param in update_parameters if not 'encoder' in name], 'lr': self.lr * 5, 'weight_decay': 1e-4}
        # ], eps=1e-8, betas=(0.9, 0.999))

        total_steps = len(self.train_dataloader) * self.num_epochs
        x1=len(self.train_dataloader)
        warmup_steps = math.ceil(total_steps * 0.05)
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps, last_epoch=-1)
        if self.fp16:
            self.model, optimizer, self.train_dataloader, self.valid_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.valid_dataloader, scheduler
            )

        train_history = collections.defaultdict(list)
        valid_history = collections.defaultdict(list)
        for epoch in range(1, self.num_epochs + 1):
            train_metric = self.train(self.train_dataloader, optimizer, criterion=self.criterion, scheduler=scheduler)
            valid_metric = self.evaluate(self.valid_dataloader, criterion=self.criterion)
            print(f'Epoch {epoch}: Train metric={train_metric}\nvalid metric={valid_metric}')
            logger.info(f'Epoch {epoch}: Train metric={train_metric}\nvalid metric={valid_metric}')

            for k, v in train_metric.items():
                train_history[k].append(v)
            for k, v in valid_metric.items():
                valid_history[k].append(v)
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            self.save_best_model(self.model_save_path, curr_metric=valid_metric[self.monitor])
        history = {'train_history': train_history, 'valid_history': valid_history}
        json.dump(history,
                  open(os.path.join(self.model_save_path, f'{self.model_name}_history.json'), 'w', encoding='utf-8'))
        return history


