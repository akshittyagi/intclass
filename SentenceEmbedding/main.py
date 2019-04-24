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

    # epoch = 10
    # # sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=epoch, batch_size=128, debug=True)
    # # sentence_embedder.train(data_train, data_dev, model_type='recurrent_bn')
    # # ac, f_mac, f_mic = sentence_embedder.test(data_test, model_type='recurrent_bn')
    # with open('epochs_gridsearch.csv', 'w') as f:
    #     f.writelines('Epochs, Accuracy, F1_macro, F1_micro\n')
    #     for ep in range(4, 15):
    #         sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=ep, batch_size=128, debug=True)
    #         print('Training for {} epochs'.format(ep))
    #         sentence_embedder.train(data_train, data_dev, model_type='recurrent_bn')
    #         ac, f_mac, f_mic = sentence_embedder.test(data_test, model_type='recurrent_bn')
    #         os.remove('bn_av_sent_emb_3_layer_glove.MODEL')
    #         f.write('{}, {}, {}, {}\n'.format(ep, ac, f_mac, f_mic))
    #         print('File removed')

    epochs = 5
    # sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=epoch, batch_size=128, debug=True)
    # sentence_embedder.train(data_train, data_dev, model_type='recurrent_bn')
    # sentence_embedder.test(data_test, model_type='recurrent_bn')
    
    # ep_list = [5, 10, 15]
    # cl_list = [0, 0.5, 1, 5, 10, 50, 100]
    ep_list = [6]
    cl_list = [0]

    # for ep in [5, 10, 15]:
    #     for cl in [0, 0.5, 1, 5, 10, 50, 100]:
    #         sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=ep, batch_size=128, debug=True)
    #         print('Training for {} epochs with clipping parameter {}'.format(ep, cl))
    #         sentence_embedder.train(data_train, data_dev, clip=cl, model_type='recurrent_bn')
    #         sentence_embedder.test(data_test, model_type='recurrent_bn')
    #         os.remove('bn_av_sent_emb_3_layer_glove.MODEL')
    #         print('File removed')

    sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=epochs, batch_size=128, debug=True)
    sentence_embedder.train_bert(data_test)
    # with open('epoch_clipping_gridsearch.csv', 'w') as f:
    #     f.writelines('Epochs, Clip, Accuracy, F1_macro, F1_micro')
    #     for ep in [5, 10, 15]:
    #         for cl in [0, 0.5, 1, 5, 10, 50, 100]:
    #             sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=ep, batch_size=128, debug=True)
    #             print('Training for {} epochs with clipping parameter {}'.format(ep, cl))
    #             sentence_embedder.train(data_train, data_dev, clip=cl, model_type='recurrent_bn')
    #             ac, f_mac, f_mic = sentence_embedder.test(data_test, model_type='recurrent_bn')
    #             os.remove('bn_av_sent_emb_3_layer_glove.MODEL')
    #             f.writelines(ep, cl, ac, f_mac, f_mic)
    #             print('File removed')

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
