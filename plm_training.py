from trainers import PLMTrainer
from models.plm_models import Prott5
from dataset import make_plm_dataloader
from transformers import T5Tokenizer
import torch
import os
import numpy as np
import random
import json

# 设置随机库的种子
seed = 21
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个CUDA设备
np.random.seed(seed)
random.seed(seed)
# 为了确保CUDA的确定性行为，可以设置以下两个配置，但这可能会牺牲一些性能
# 注意：这可能会影响性能
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义参数
train_file = 'data/train.csv'
valid_file = 'data/valid.csv'
test_file = 'data/test.csv'

# 模型相关
save_path = 'saved-ProTCR-MHC1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_path = './prot_t5_xl_uniref50'
model = Prott5.from_pretrained(pretrained_path, num_labels=2,dropout_rate=0.1)

model.to(device)

model_name = type(model).__name__
model_save_path = os.path.join(save_path,model_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


# 训练相关
lr = 2e-5
num_epochs = 20
batch_size = 64 # 64分之一 best

tokenizer = T5Tokenizer.from_pretrained(pretrained_path)

train_dataloader, valid_dataloader, test_dataloader = make_plm_dataloader(
    train_file, valid_file, test_file, tokenizer=tokenizer, batch_size=batch_size)


trainer = PLMTrainer(model, train_dataloader, valid_dataloader,
                 lr=lr, num_epochs=num_epochs, batch_size=batch_size, save_path=save_path,
                 model_name=model_name, monitor='f1', average='binary',
                 device=device, fp16=True)
trainer.training()

# 评估
model_save_path = os.path.join(save_path, model_name)
trainer.model = Prott5.from_pretrained(model_save_path)
trainer.model.to(device)
test_metrics = trainer.evaluate(test_dataloader, is_test=True)
print(f'test_metrics: {test_metrics}')
test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
# json.dump(test_metrics, open(os.path.join('model_metrics', f'{model_name}.json'),'w'))
