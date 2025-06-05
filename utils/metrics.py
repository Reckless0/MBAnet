def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
           FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
           FN += 1
        if y_true[i] == 0 and y_pred[i] == 0:
           TN += 1

    return TP, FP, FN, TN