
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial import distance



def euclideandetector(data, sample, users):

    train, test, impostor = split_data(data, sample, users)
    mean_vector = []
    impostor_user = []
    genuine_user = []

    for i in range(0, train.shape[0], sample):
        mean_vector.append(train.iloc[i:i + sample].mean())

    for i in range(0, impostor.shape[0], int((impostor.shape[0]) / len(users))):
        impostor_user.append(impostor.iloc[i:i + int((impostor.shape[0]) / len(users))])

    for i in range(0, test.shape[0], int((test.shape[0]) / len(users))):
        genuine_user.append(test.iloc[i:i + int((test.shape[0]) / len(users))])

    impostor_score = evaluateScore(mean_vector, impostor_user, users)
    genuine_score = evaluateScore(mean_vector, genuine_user, users)

    fpr, ipr, tpr, eer, threshold = evaluateEER(genuine_score, impostor_score,users)

    return fpr,ipr,tpr,threshold,eer


def visualroc(fpr, tpr, n, color, sample_size):
    plt.figure(figsize=(8,6))
    c = -1
    for i in n:
        c +=1
        roc_auc=auc(fpr[i],tpr[i])
        plt.plot(fpr[i],tpr[i],color=color[c],label='ROC Curve (area ='+'{:3f}'.format(roc_auc)+') '+
                'user '+'{}'.format(i+1))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Euclidean Detector : Receiver Operating Characteristic for '+'{}'.format(len(n))+' users  (N= {})'.format(sample_size))
        plt.legend(loc="lower right")
    plt.show()

def visualdet(fpr, ipr, n, t,sample_size,color):
    plt.figure(figsize=(8,6))
    c=-1
    for i in n:
        c+=1
        plt.plot(t[i],fpr[i], color=color[c], label='False Positive Rate (User '+'{}'.format(i)+')')
        plt.plot(t[i],ipr[i], color=color[c], label='Impostor Pass Rate(User '+'{}'.format(i)+')')
        plt.xlim([0.0, max(t[i])])
        plt.ylim([0.0, 1.05])
        plt.ylabel('Error Rate')
        plt.xlabel('Threshold')
        plt.title('Euclidean Detector :Detection Error Tradeoff for' + '{}'.format(len(n)) + ' users (N= {})'.format(sample_size))
        plt.legend(loc="lower right")
    plt.show()

def evaluateEER(genuine, impostor,users):
    eer_list = []
    fprl = []
    iprl = []
    tprl = []
    threshold_list = []
    for i in range(len(users)):
        labels = [0] * len(genuine[i]) + [1] * len(impostor[i])
        scores = list(genuine[i]) + list(impostor[i])
        fpr, tpr, threshold = roc_curve(labels, scores)
        ipr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute(ipr - fpr))]
        eer = fpr[np.nanargmin(np.absolute((ipr - fpr)))]
        eer_list.append(eer)
        fprl.append(fpr)
        iprl.append(ipr)
        tprl.append(tpr)
        threshold_list.append(threshold)

    return fprl, iprl, tprl, eer_list, threshold_list




def split_data(data,sample,users):
    train = pd.DataFrame()
    test = pd.DataFrame()
    impostor = pd.DataFrame()

    for user in users:
        temp = data.loc[data.subject == user]
        train = train.append(temp[:sample])
        test = test.append(temp[sample:])

    for user in users:
        temp = test.loc[test.subject != user, 'H.period':'H.Return']
        impostor = impostor.append(temp)

    return train.iloc[:, 3:], test.iloc[:, 3:], impostor


def evaluateScore(template, score_list, users):
    score = []
    total = []
    norm_score = []
    for i in range(len(users)):
        for j in range(len(score_list[i])):
            norm_score.append(np.linalg.norm(np.array(score_list[i].iloc[j])) * np.linalg.norm(np.array(template[i])))
            score.append(distance.euclidean(score_list[i].values[j], template[i]) / norm_score[j])

    for i in range(0, len(score), int(len(score) / len(users))):
        total.append(score[i:i + int(len(score) / len(users))])

    return total

