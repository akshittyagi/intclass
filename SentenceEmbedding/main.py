import os

import sklearn

from Data import data_loader
from Model import SentenceEmbedder

if __name__ == "__main__":
    #### ATIS ####
    # dataset_loc = "../DataSets/ATIS/"
    # data_train, data_dev, data_test = data_loader(dataset_loc, dataset='atis')
    # sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=20)
    # sentence_embedder.train(data_train, data_dev, model_type='feed_forward')
    # sentence_embedder.test(data_test, model_type='feed_forward')

    #### FB ####
    dataset_loc = "../DataSets/top-dataset-semantic-parsing/"
    data_train, data_dev, data_test = data_loader(dataset_loc, dataset='fb')
    epoch = 4
    # sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=epoch, batch_size=128, debug=True)
    # sentence_embedder.train(data_train, data_dev, model_type='recurrent_bn')
    # sentence_embedder.test(data_test, model_type='recurrent_bn')

    for ep in [5, 10, 15]:
        for cl in [0, 0.5, 1, 5, 10, 50, 100]:
            sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=ep, batch_size=128, debug=True)
            print('Training for {} epochs with clipping parameter {}'.format(ep, cl))
            sentence_embedder.train(data_train, data_dev, clip=cl, model_type='recurrent_bn')
            sentence_embedder.test(data_test, model_type='recurrent_bn')
            os.remove('bn_av_sent_emb_3_layer_glove.MODEL')
            print('File removed')

    # dataset_loc = "../DataSets/top-dataset-semantic-parsing/"
    # data_train, data_dev, data_test = data_loader(dataset_loc, dataset='fb')
    # sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=40, batch_size=128)
    # sentence_embedder.train(data_train, data_dev, model_type='feed_forward')
    # sentence_embedder.test(data_test, model_type='feed_forward')

    # dataset_loc = "../DataSets/top-dataset-semantic-parsing/"
    # data_train, data_dev, data_test = data_loader(dataset_loc, dataset='fb')
    # sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=40, batch_size=128)
    # sentence_embedder.train(data_train, data_dev, model_type='feed_forward_bn')
    # sentence_embedder.test(data_test, model_type='feed_forward_bn')
