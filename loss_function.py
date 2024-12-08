import numpy as np
import tensorflow.keras.backend as K

def weighted_categorical_cross_entropy(weights: np.ndarray):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    :param weights: a list of every class' weight

    :return: a weighted categorical cross entropy loss function

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_cross_entropy(weights)
        model.compile(loss=loss, optimizer='adam')
    """

    weights = K.variable(weights)

    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0) error
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate cross entropy
        loss = y_true * K.log(y_pred) * weights
        # Sum over classes
        loss = -K.sum(loss, axis=-1)
        return loss

    return loss_fn
