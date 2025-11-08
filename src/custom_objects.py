import tensorflow.keras.backend as K
def mean_iou(y_true, y_pred,smooth=1):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)



def tpr(y_true, y_pred, smooth=1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    # Convert probabilities/logits to binary 0/1
    y_pred_bin = K.round(K.clip(y_pred, 0, 1))

    tp = K.sum(y_true * y_pred_bin)                    # True Positives
    p = K.sum(y_true)                                  # Actual Positives

    return (tp + smooth) / (p + smooth)

def fpr(y_true, y_pred, smooth=1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_pred_bin = K.round(K.clip(y_pred, 0, 1))

    # False Positive: predicted 1 but actually 0
    fp = K.sum((1 - y_true) * y_pred_bin)

    # True Negative: predicted 0 and actually 0
    tn = K.sum((1 - y_true) * (1 - y_pred_bin))

    return (fp + smooth) / (fp + tn + smooth)

def dice_loss(y_true,y_pred):
    return 1-dice_coefficient(y_true,y_pred)

