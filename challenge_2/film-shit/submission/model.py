
import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.models = [tf.keras.models.load_model(os.path.join(path, f"SubmissionModel{i+1}")) for i in range(6)]

    def predict(self, X):

        outputs = [self.models[i].predict(X) for i in range(5)]
        out = tf.argmax(sum(outputs), axis=-1)

        return out
