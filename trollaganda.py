"""
Main python script to run the program
"""

from embedding import ReadFile, PreProcess, PrepareEmbedding
from model import TheModel
from predictor import Predictor

import config

readfile = ReadFile(path="./output.csv", split=config.SPLITDATA)
readfile.readfile()
readfile.distribution_plot()

pre_proc = PreProcess(data=readfile.data, textfield="message")
pre_proc.process_text()
pre_proc.see_data_head()

embed = PrepareEmbedding(
    X=readfile.data.message,
    Y=readfile.data.isTroll,
    embedded_path="./GoogleNews-vectors-negative300.bin.gz"
)
embed.print_info()
embed.load_word_2_vec()
embed.train()
embed.release_pre_trained()

model = TheModel()
model.conv_net(
    embeddings=embed.train_embedding_weights,
    max_sequence_length=config.MAXSEQLENGTH,
    num_words=len(embed.train_word_index) + 1,
    embedding_dim=config.EMBEDDINGDIM,
    trainable=False
)
model.train_model(
    traincnndata=embed.train_cnn_data,
    Y_train=embed.Y_train,
    epochs=config.EPOCHS,
    batch_size=config.BATCHSIZE
)
model.evaluate_model(test_cnn_data=embed.test_cnn_data, Y_test=embed.Y_test)

# TODO argparse, make the script above configurable(+glove)


# Prediction example

predictor = Predictor(model = model, embed = embed)
predictor.predict(messages=["Some message @ https://google.com where @liran23 said he wants to read it", "Yossi is the man and he loves to post links online"])