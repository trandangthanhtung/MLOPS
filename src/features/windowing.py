import numpy as np

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)


def sliding_window_inference(latest_window):
    """
    latest_window: (SEQ_LEN, num_features)
    """
    return np.expand_dims(latest_window, axis=0)
