from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import numpy as np



def Performance(Y_train,Y_pred1):
    
    cnf_matrix= confusion_matrix(Y_train,Y_pred1)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)    
        
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # detection_rate
    detection_rate=TN/(TN+TP+FP+FN)
    #kappa
    n=len(Y_train)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    k=(ko-ke)/(1-ke)
    ppv=sum(PPV)/len(PPV)
    NPV=sum(NPV)/len(NPV)
    
    Accuracy=sum(ACC)/len(ACC)
   # print ('Accuracy : ', Accuracy1)
    Sensitivity=sum(TPR)/len(TPR)
   # print ('Sensitivity : ', Sensitivity1)
    Specificity=sum(TNR)/len(TNR)
   # print ('Specificity : ', Specificity1)
    precision=sum(PPV)/len(PPV)
   # print ('Precision : ', precision1)
    f1_score=(2*precision*Sensitivity)/(precision+Sensitivity)
   # print ('f1_score : ', f1_score1)
    Recall =sum(TPR)/len(TPR)
    return cnf_matrix, Accuracy,Sensitivity,Specificity,precision,f1_score,Recall