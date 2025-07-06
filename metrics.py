from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc, classification_report
import numpy as np

def calculate_multiclass_micro_auc(true_labels, pred_probs, labels=None):
    pred_probs = np.array(pred_probs)
    true_labels = np.array(true_labels)
    num_classes = pred_probs.shape[1]

    if num_classes == 2:
        auc = roc_auc_score(true_labels, pred_probs[:,1])
    else:
        # 多分类计算micro auc
        micro_auc_scores = []
        if labels is None:
            classes = list(range(num_classes))
        else:
            classes = labels
        for class_idx in classes:
            binary_true_labels = (true_labels == class_idx).astype(int)
            binary_probs = pred_probs[:, class_idx]
            micro_auc = roc_auc_score(binary_true_labels, binary_probs)
            micro_auc_scores.append(micro_auc)
        auc = np.mean(micro_auc_scores)
    return auc



def cal_acc_precision_recall_f1(real, pred, pred_probs=None, average='micro', is_test=False):
    # 计算预测结果的精确率、召回率和F1值
    p = precision_score(real, pred, average=average, )
    r = recall_score(real, pred, average=average)
    f1 = f1_score(real, pred, average=average)
    acc = accuracy_score(real, pred)

    auc = None
    try:
        if pred_probs is not None:
            # 计算auc
            auc = calculate_multiclass_micro_auc(real, pred_probs)
    except:
        pass
    # 返回包含精确率、召回率和F1值的字典
    metric =  {'acc': acc, 'precision': p.round(4), 'recall': r.round(4), 'f1': f1.round(4), 'auc': auc}
    if is_test:
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        report_dict = classification_report(real, pred, output_dict=True)
        metric['report'] = report_dict
        # 混淆矩阵
        conf_matrix = confusion_matrix(real, pred)
        tn,fp,fn,tp = confusion_matrix(real, pred).ravel()
        print(tn,fp,fn,tp)
        metric['confusion_matrix'] = conf_matrix
    return metric

