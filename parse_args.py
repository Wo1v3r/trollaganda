
import argparse

def parseArgs():
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

    return args

