import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    
    prediction_shape = prediction.shape[0]
    positives = np.sum(ground_truth)
    negatives = np.sum(ground_truth == False)
    assert negatives == prediction_shape - positives, "positives + negatives not correct size"

    TP = np.sum(np.logical_and(prediction == 1, ground_truth == 1))
    FP = np.sum(np.logical_and(prediction == 1, ground_truth == 0))
    FN = np.sum(np.logical_and(prediction == 0, ground_truth == 1))

    precision = TP / (FP + TP)
    recall = TP / (FN + TP)
    f1 = (2 * precision * recall) / (precision + recall)
    accuracy = np.mean(prediction == ground_truth)
    

    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = np.mean(prediction == ground_truth)
    return accuracy
