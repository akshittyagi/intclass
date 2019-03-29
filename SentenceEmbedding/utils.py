from gensim.models.callbacks import CallbackAny2Vec
import numpy as np

class EpochLogger(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

class MonitorLossLogger(CallbackAny2Vec):

    def on_epoch_end(self, model):
        print("Model Loss: ", model.get_latest_training_loss())

def fb_top_intent(intents):
    return intents.split()[0][1:]

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) * 1.0 / len(y_true)