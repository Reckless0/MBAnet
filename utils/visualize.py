import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from matplotlib.pyplot import MultipleLocator

"""
Draw ROC and PR curve for every fold.
"""
def draw_fold_ROC_PR(score_tensor, label_onehot, y_real, y_score, aucs, k_th_fold, ax_roc, ax_pr, opt, optimal_metrics='ROC'):

    fpr, tpr, thres_roc = roc_curve(label_onehot[:, 1], score_tensor[:, 1])
    roc_auc = auc(fpr, tpr)

    ax_roc.plot(
        fpr,
        tpr,
        lw=1,
        label='ROC fold %s, AUC = %.3f'%(k_th_fold + 1, roc_auc),
    )

    y_real.append(label_onehot[:, 1])
    y_score.append(score_tensor[:, 1])
    aucs.append(roc_auc)

    precision, recall, thres_pr = precision_recall_curve(label_onehot[:, 1], score_tensor[:, 1], )
    average_precision = average_precision_score(label_onehot[:, 1], score_tensor[:, 1], average="micro")

    ax_pr.plot(
        recall,
        precision,
        label='PR Curve fold %d, AUC = %.3f'%(k_th_fold + 1, average_precision),
        alpha=0.3,
        lw=1,
    )

    y_real.append(label_onehot[:, 1])
    y_score.append(score_tensor[:, 1])

    if optimal_metrics == 'ROC':
        optimal_threshold_index = np.argmax(tpr - fpr)
        optimal_threshold = thres_roc[optimal_threshold_index] if thres_roc[optimal_threshold_index] < 0.5 else 0.5

    elif optimal_metrics == 'PR':
        f_score = []
        for i in range(len(precision)):
            if precision[i] == recall[i] == 0:
                continue 
            else:
                beta = opt.weight_pr_ratio
                beta2 = beta**2
                f_score.append(((1+beta2) * precision[i] * recall[i]) / (beta2*precision[i] + recall[i])) 
        
        optimal_threshold_index = np.argmax(f_score)
        optimal_threshold = thres_pr[optimal_threshold_index] if thres_pr[optimal_threshold_index] < 0.5 else 0.5
        print("optimal_threshold:", optimal_threshold)
    else:
        raise ValueError("Not available value for argument:'optimal_metrics'.")

    return y_real, y_score, aucs, optimal_threshold


def draw_total_ROC_PR(y_real, y_score, aucs, ax_roc, ax_pr):
    y_real = np.concatenate(y_real)
    y_score = torch.tensor(np.concatenate(y_score))

    fpr, tpr, thres_roc = roc_curve(y_real, y_score, pos_label=1)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    print("fpr:", fpr)
    print("tpr:", tpr)
    print("thres_roc:", thres_roc)
    AUC = auc(fpr, tpr)
    std_auc = np.std(aucs)
    ax_roc.plot(fpr, tpr, color='b',
            label=r'Mean ROC Curve',
            lw=2, alpha=.8)

    ax_roc.set(
        xlim=[0, 1.05],
        ylim=[0, 1.05],
        title="Receiver operating characteristic (AUC = %0.3f $\pm$ %0.3f)" % (AUC, std_auc),
        xlabel='False positive rate',
        ylabel='True positive rate',
    )
    ax_roc.legend(loc="lower right")

    precision, recall, thres_pr = precision_recall_curve(y_real, y_score)
    print("precision:", precision)
    print("recall:", recall)
    print("thres_pr:", thres_pr)
    ax_pr.plot(recall, precision, color='b',
             label=r'Mean PR Curve',
             lw=2, alpha=.8)
    ax_pr.set(
        xlim=[0, 1.05],
        ylim=[0, 1.05],
        xlabel='Recall',
        ylabel='Precision',
        title='Precision-Recall (AP = %.3f)' % (average_precision_score(y_real, y_score)),
    )

    ax_pr.legend(loc="lower right")

    return ax_roc, ax_pr


def draw_test_ROC_PR(score, label, ax_roc, ax_pr, hospital=None, fold_num=None):
    color_maps =['#BB9727',
                '#54B345',
                '#32B897',
                '#05B9E2',
                '#8983BF']
    if hospital:
        # fpr, tpr, _ = roc_curve(label_onehot[:, 1], score_tensor[:, 1])
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)

        ax_roc.plot(
            fpr, tpr,
            # color='b',
            lw=2,
            # alpha=.8,
            label=str(hospital)+" AUC = %.3f" % (roc_auc),
        )

        # ax_roc.set(
        #     xlim=[0, 1.05],
        #     ylim=[0, 1.05],
            # title="Receiver operating characteristic (AUC = %.3f)" % (roc_auc),
            # xlabel='Sensitivity',
            # ylabel='1-Specificity',
        # )
        # ax_roc.set_title("Receiver operating characteristic (AUC = %.3f)" % (roc_auc), fontsize=18)
        # ax_roc.set_xlabel('Sensitivity', fontdict={'size': 14})
        # ax_roc.set_ylabel('1-Specificity', fontdict={'size': 14})
        # ax_roc.set_title("Receiver Operating Characteristic Curve")
        ax_roc.set_xlabel('1-Specificity', fontdict={'size':18})
        ax_roc.set_ylabel('Sensitivity', fontdict={'size':18})
        ax_roc.autoscale(tight=True)
        ax_roc.set_xlim([0, 1.02])
        ax_roc.set_ylim([0, 1.02])
        ax_roc.xaxis.set_major_locator(MultipleLocator(0.1))  # 修改轴刻度显示间隔为0.1的倍数
        ax_roc.yaxis.set_major_locator(MultipleLocator(0.1))
        # ax_roc.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
        # ax_roc.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
        # ax_roc.set_color()

        ax_roc.legend(loc="lower right", fontsize=18)

        # precision, recall, _ = precision_recall_curve(label_onehot[:, 1], score_tensor[:, 1], )
        # average_precision = average_precision_score(label_onehot[:, 1], score_tensor[:, 1], average="micro")
        precision, recall, _ = precision_recall_curve(label, score, )
        average_precision = average_precision_score(label, score, average="micro")

        ax_pr.plot(
            recall,
            precision,
            # color='b',
            label=str(hospital)+"AUC = %.3f" % (average_precision),
            lw=2,
            alpha=.5
        )

        # ax_pr.set(
        #     xlim=[0, 1.05],
        #     ylim=[0, 1.05],
        #     xlabel='Recall',
        #     ylabel='Precision',
        #     title='Precision-Recall (AP = %.3f)' % (average_precision),
        # )
        # ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel('Recall', fontdict={'size': 18})
        ax_pr.set_ylabel('Precision', fontdict={'size': 18})
        ax_pr.autoscale(tight=True)
        ax_pr.set_xlim([0, 1.02])
        ax_pr.set_ylim([0, 1.02])
        ax_pr.xaxis.set_major_locator(MultipleLocator(0.1))  # 修改轴刻度显示间隔为0.1的倍数
        ax_pr.yaxis.set_major_locator(MultipleLocator(0.1))

        ax_pr.legend(loc="lower right", fontsize=18)
    
    else:
        # fpr, tpr, _ = roc_curve(label_onehot[:, 1], score_tensor[:, 1])
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)

        if fold_num !=None:
            ax_roc.plot(
                fpr, tpr,
                color=color_maps[fold_num],
                lw=1, #2,
                # alpha=.5,
                label=f"Expert-{fold_num+1} AUROC = %.3f" % (roc_auc),
            )
        else:
            ax_roc.plot(
                fpr, tpr,
                color='#F27970',
                lw=2,
                # alpha=.8,
                label="Ensembled AUROC = %.3f" % (roc_auc),
            )

        # 添加x轴每经过0.2的垂直辅助线
        for i in np.arange(0.2, 1.2, 0.2):
            ax_roc.axvline(x=i, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
        # 添加y轴每经过0.2的水平辅助线
        for i in np.arange(0.2, 1.2, 0.2):
            ax_roc.axhline(y=i, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)



        # ax_roc.set(
        #     xlim=[0, 1.05],
        #     ylim=[0, 1.05],
            # title="Receiver operating characteristic (AUC = %.3f)" % (roc_auc),
            # xlabel='Sensitivity',
            # ylabel='1-Specificity',
        # )
        # ax_roc.set_title("Receiver operating characteristic (AUC = %.3f)" % (roc_auc), fontsize=18)
        # ax_roc.set_xlabel('Sensitivity', fontdict={'size': 14})
        # ax_roc.set_ylabel('1-Specificity', fontdict={'size': 14})
        # ax_roc.set_title("Receiver Operating Characteristic Curve")
        ax_roc.set_xlabel('1-Specificity', fontdict={'size':18})
        ax_roc.set_ylabel('Sensitivity', fontdict={'size':18})
        ax_roc.autoscale(tight=True)
        ax_roc.set_xlim([0, 1.02])
        ax_roc.set_ylim([0, 1.02])
        ax_roc.xaxis.set_major_locator(MultipleLocator(0.1))  # 修改轴刻度显示间隔为0.1的倍数
        ax_roc.yaxis.set_major_locator(MultipleLocator(0.1))
        # ax_roc.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
        # ax_roc.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
        # ax_roc.set_color()

        ax_roc.legend(loc="lower right", fontsize=18)

        # precision, recall, _ = precision_recall_curve(label_onehot[:, 1], score_tensor[:, 1], )
        # average_precision = average_precision_score(label_onehot[:, 1], score_tensor[:, 1], average="micro")
        precision, recall, _ = precision_recall_curve(label, score, )
        average_precision = average_precision_score(label, score, average="micro")
        
        if fold_num != None:
            ax_pr.plot(
                recall,
                precision,
                color=color_maps[fold_num],
                label=f"Expert-{fold_num+1} AUPR = %.3f" % (average_precision),
                lw=1, #2,
                # alpha=.5
            )
        else:
            ax_pr.plot(
                recall,
                precision,
                color='#F27970',
                label="Ensembled AUPR = %.3f" % (average_precision),
                lw=2,
                # alpha=.8
            )
        
        # 添加x轴每经过0.2的垂直辅助线
        for i in np.arange(0.2, 1.2, 0.2):
            ax_pr.axvline(x=i, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
        # 添加y轴每经过0.2的水平辅助线
        for i in np.arange(0.2, 1.2, 0.2):
            ax_pr.axhline(y=i, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)

        # ax_pr.set(
        #     xlim=[0, 1.05],
        #     ylim=[0, 1.05],
        #     xlabel='Recall',
        #     ylabel='Precision',
        #     title='Precision-Recall (AP = %.3f)' % (average_precision),
        # )
        # ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel('Recall', fontdict={'size': 18})
        ax_pr.set_ylabel('Precision', fontdict={'size': 18})
        ax_pr.autoscale(tight=True)
        ax_pr.set_xlim([0, 1.02])
        ax_pr.set_ylim([0, 1.02])
        ax_pr.xaxis.set_major_locator(MultipleLocator(0.1))  # 修改轴刻度显示间隔为0.1的倍数
        ax_pr.yaxis.set_major_locator(MultipleLocator(0.1))

        ax_pr.legend(loc="lower right", fontsize=18)

    return ax_roc, ax_pr