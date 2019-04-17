"""
Main python script to run the program
"""

import argparse
from embedding import ReadFile, PreProcess, PrepareEmbedding
from model import TheModel
from predictor import Predictor

import config

parser = argparse.ArgumentParser(description="Machine learning for detecting troll tweets")
loaded = parser.add_mutually_exclusive_group()
loaded.add_argument("-t", "--train", action="store_true", help="Train the data-sets and model")
parser.add_argument("-p", "--predict", nargs="+", help="Predict given strings")
loaded.add_argument("-l", "--load_model",
                    nargs="?", default="./model", help="Load the model from a given path, default path is \"./model\"")
parser.add_argument("--split", nargs="?", type=int,
                    default=10000, help="Split the data-sets to a smaller one, default is 10000")
parser.add_argument("--embedding_dim", nargs="?",
                    default=300, type=int, help="The embeddings vector dimensions, default is 300")
parser.add_argument("--max_vocabulary", nargs="?", default=175303, type=int, help="The maximum vocabulary size")
parser.add_argument("--max_sequence", nargs="?",
                    default=200, type=int, help="The maximum sequence length, default is 200")
parser.add_argument("-b", "--batch_size", nargs="?", default=256, type=int, help="Batch size, default is 256")
parser.add_argument("-e", "--epochs", nargs="?", default=3, type=int, help="Epochs to train, default is 3")
parser.add_argument("-i", "--input", nargs="?",
                    default="./output.csv", help="Input data path, default is \"./output.csv\"")
parser.add_argument("-s", "--save_model", nargs="?", default="./model", help="Save the model, default is \"./model\"")

trained = parser.add_mutually_exclusive_group()
trained.add_argument("--word2vec", action="store_true", help="Use word2vec google's pre-trained words, used by default")
trained.add_argument("--glove", action="store_true", help="Use gloVe pre-trained words")
parser.add_argument("--plot", action="store_true", help="Print plots and some data in the process")


args = parser.parse_args()
print(args)

model = None
embed = None

if args.train:
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
    predictor.predict(
        messages=[
            "Some message @ https://google.com where @liran23 said he wants to read it",
            "Yossi is the man and he loves to post links online"
        ]
    )

    predictor.predict(messages=args.predict)
