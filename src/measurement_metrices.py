import numpy as np
import math as m
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.preprocessing import LabelBinarizer


#https://bitcoden.com/answers/calculate-sklearnroc_auc_score-for-multi-class
#https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
#MCC link: https://blester125.com/blog/rk.html



def mcc(x,y):
    cm = confusion_matrix(x, y)
    samples = np.sum(cm)
    correct = np.trace(cm)
    y = np.sum(cm, axis=1, dtype=np.float64)
    x = np.sum(cm, axis=0, dtype=np.float64)
    cov_x_y = correct * samples - np.dot(x, y)
    cov_y_y = samples * samples - np.dot(y, y)
    cov_x_x = samples * samples - np.dot(x, x)
 
    denom = np.sqrt(cov_x_x * cov_y_y)
    denom = denom if denom != 0.0 else 1.0
    return cov_x_y / denom


def mcc_each(n_classes,tp,tn,fp,fn):
    MCC=[]
    for i in range(n_classes):
        TP=tp[i]
        TN=tn[i]
        FP=fp[i]
        FN=fn[i]
        dominator = m.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if dominator==0:
            mcc = 0
        else:
            mcc=(TP*TN-FP*FN)/dominator
        MCC.append(mcc)
    return MCC

    
def performance(y_true, y_pred):
    l=[]
    cnf_matrix = confusion_matrix(y_true, y_pred)
    fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    tp = np.diag(cnf_matrix)
    tn = cnf_matrix.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)
    #Value counts of predictions
    labels, cnt = np.unique(
        y_true,
        return_counts=True)

    n_classes = len(labels)
    
    acc = (tp+tn)/(tn+tp+fn+fp)
    sen = tp/(tp+fn)
    spec = tn/(fp+tn)
   
    MCC=mcc_each(n_classes,tp,tn,fp,fn)   
         
    l.append([tp,tn,fp,fn])
    return acc,sen,spec,MCC,tp,tn,fp,fn    
    
def roc_auc(y_true, y_pred, y_score):
    tpr_final=[]
    fpr_final=[]
    auc_final=[]
    
    #Value counts of predictions
    labels, cnt = np.unique(
        y_true,
        return_counts=True)

    n_classes = len(labels)

    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for label_it, label in enumerate(labels):
        fpr[label], tpr[label], _ = roc_curve(
            (y_true == label).astype(int), 
            y_score[:, label_it])

        roc_auc[label] = auc(fpr[label], tpr[label])
        tpr_final.append(tpr[label])
        fpr_final.append(fpr[label])
        auc_final.append(roc_auc[label])
       
    return tpr_final,fpr_final,auc_final  
    
    


def class_report_final(y_true, y_pred, y_score=None, average='macro'):
   
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = label_binarize()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_true,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)



    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

    return   roc_auc["avg / total"] , fpr["macro"],tpr["macro"] 
