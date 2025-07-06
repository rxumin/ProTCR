from trainers import PLMTrainer
from models.plm_models import Prott5
from dataset import make_plm_dataloader
from transformers import T5Tokenizer
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

model = Prott5.from_pretrained('./saved-mer/Prott5')
# model = Prott5.from_pretrained('./saved-MHC2/Prott5')

# MHC1
test_file = './data/nettcr/test.csv'
# MHC2
# test_file = 'data/MHC2/1/test.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
max_length = 200
model.to(device)
model.eval()

tokenizer = T5Tokenizer.from_pretrained('./prot_t5_xl_uniref50')
_, _, test_dataloader = make_plm_dataloader(
    test_file, test_file, test_file, tokenizer=tokenizer, batch_size=batch_size)

real_labels = []
pred_labels = []
pred_probs = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        # 将数据和标签移动到指定设备上
        batch = {k: batch[k].to(device) for k in batch}
        output = model(**batch)
        _pred_labels = torch.argmax(output.logits, dim=-1)
        _pred_probs = torch.softmax(output.logits, dim=-1)[:,1]
        real_labels += batch['labels'].cpu().numpy().tolist()
        pred_labels += _pred_labels.cpu().numpy().tolist()
        pred_probs += _pred_probs.cpu().numpy().tolist()

# np.save('result/nettcr/real_labels-mr2-b.npy', real_labels)
# np.save('result/nettcr/pred_probs-mr2-b.npy', pred_probs)

df = pd.read_csv(test_file, sep='\t')
df['plabels'] = pred_labels
# df.to_excel('./data/MHC2/infer_data.xlsx', index=False)

# 创建一个图形
plt.figure(figsize=(8, 6))

fpr, tpr, _ = roc_curve(real_labels, pred_probs)
roc_auc = roc_auc_score(real_labels, pred_probs)
# 绘制 ROC 曲线
plt.plot(fpr, tpr, label=' (AUC = {:.4f})'.format(roc_auc))
# 添加标签和图例
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# plt.savefig('./result/roc-mhc2.jpeg', dpi=300)
plt.show()
