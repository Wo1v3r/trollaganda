"""
Main python script to run the program
"""

from embedding import ReadFile, PreProcess, PrepareEmbedding
from model import TheModel
from predictor import Predictor

import config


def setup(args):
    model = None
    # embed = None

    readfile = ReadFile(path=args.input, split=args.split)
    readfile.readfile()
    if args.plot:
        readfile.distribution_plot()


    pre_proc = PreProcess(data=readfile.data, textfield="message")
    pre_proc.process_text()
    if args.plot:
        pre_proc.see_data_head()
    # TODO add opional paths for pre-train word sets

    embedded_path = "./GoogleNews-vectors-negative300.bin.gz" if args.word2vec else "./glove.txt"
    embed = PrepareEmbedding(
        X=readfile.data.message,
        Y=readfile.data.isTroll,
        embedded_path=embedded_path
    )
    if args.plot:
        embed.print_info()
    if args.word2vec:
        embed.load_word_2_vec()
    else:
        embed.load_glove()
    embed.train(args.max_vocabulary)
    embed.release_pre_trained()

    if args.train:
        model = TheModel()
        model.conv_net(
            embeddings=embed.train_embedding_weights,
            max_sequence_length=args.max_sequence,
            num_words=len(embed.train_word_index) + 1,
            embedding_dim=args.embedding_dim,
            trainable=False
        )
        model.train_model(
            traincnndata=embed.train_cnn_data,
            Y_train=embed.Y_train,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        model.evaluate_model(test_cnn_data=embed.test_cnn_data, Y_test=embed.Y_test)

        if args.plot:
            model.print_accuracy_plot()
            model.print_loss_plot()

        if args.save_model:
            model.save_model(path=args.save_model)

    else:  # load model
        model = TheModel()
        path = args.load_model if args.load_model else config.LOAD_MODEL_PATH
        model.load_model(path=path)

        if args.save_model:
            model.save_model(path=args.save_model)

    if args.predict:
        # TODO split embed from predictor
        predictor = Predictor(model=model, embed=embed)
        # Example of prediction

        predictor.predict(messages=args.predict)

        return predictor
