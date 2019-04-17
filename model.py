"""
This module contains the model features
"""
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.models import model_from_json


class TheModel(object):
    """
    Model class
    """
    def __init__(self, model=None):
        self.model = model
        self.history = None

    def conv_net(
            self,
            embeddings,
            max_sequence_length,
            num_words,
            embedding_dim,
            trainable=False,
            extra_conv=True
    ):
        embedding_layer = Embedding(num_words,
                                    embedding_dim,
                                    weights=[embeddings],
                                    input_length=max_sequence_length,
                                    trainable=trainable)

        sequence_input = Input(shape=(max_sequence_length,), dtype="int32")
        embedded_sequences = embedding_layer(sequence_input)

        # Yoon Kim model (https://arxiv.org/abs/1408.5882)
        convs = []
        filter_sizes = [3, 4, 5]

        for filter_size in filter_sizes:
            l_conv = Conv1D(filters=128, kernel_size=filter_size, activation="relu")(embedded_sequences)
            l_pool = MaxPooling1D(pool_size=3)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs)

        # add a 1D convnet with global maxpooling, instead of Yoon Kim model
        conv = Conv1D(filters=128, kernel_size=3, activation="relu")(embedded_sequences)
        pool = MaxPooling1D(pool_size=3)(conv)

        if extra_conv:
            x = Dropout(0.5)(l_merge)
        else:
            # Original Yoon Kim model
            x = Dropout(0.5)(pool)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        # Finally, we feed the output into a Sigmoid layer.
        # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0)
        # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
        preds = Dense(1, activation="sigmoid")(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["acc"])
        self.model.summary()

    def train_model(self, traincnndata, Y_train, epochs, batch_size, validation_split=0.1):
        try:
            if not self.model:
                raise Exception("The model didn't initiate.")
        except Exception as e:
            print(e)
            return

        # Defining callbacks
        earlystopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=4, verbose=1)
        callbackslist = [earlystopping]
        print("Training the Model")
        self.history = self.model.fit(
            traincnndata, Y_train, epochs=epochs, callbacks=callbackslist,
            validation_split=validation_split, shuffle=True, batch_size=batch_size
        )

    def evaluate_model(self, test_cnn_data, Y_test):
        try:
            if not self.model:
                raise Exception("The model didn't initiate.")
        except Exception as e:
            print(e)
            return
        yresults = self.model.evaluate(test_cnn_data, Y_test)
        print("Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(yresults[0], yresults[1]))

    def load_model(self, path):
        json_file = open(path + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(path + ".h5")
        loaded_model.compile(loss="binary_crossentropy",
                             optimizer="adam",
                             metrics=["acc"])
        self.model = loaded_model
        print("Loaded Model from disk")

    def save_model(self, path):
        try:
            if not self.model:
                raise Exception("The model didn't initiate.")
        except Exception as e:
            print(e)
            return
        print("Saving Model to path/model.json, weights to path/model.h5")
        model_json = self.model.to_json()

        with open(path + "/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(path + "./model.h5")
        print("Model Saved")

    def print_loss_plot(self):
        try:
            if not self.history:
                raise Exception("The model didn't run.")
        except Exception as e:
            print(e)
            return
        plt.figure()
        plt.plot(self.history.history["loss"], lw=2.0, color="b", label="train")
        plt.plot(self.history.history["val_loss"], lw=2.0, color="r", label="val")
        plt.title("CNN sentiment")
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(loc="upper right")
        plt.show()

    def print_accuracy_plot(self):
        try:
            if not self.history:
                raise Exception("The model didn't run.")
        except Exception as e:
            print(e)
            return
        plt.figure()
        plt.plot(self.history.history["acc"], lw=2.0, color="b", label="train")
        plt.plot(self.history.history["val_acc"], lw=2.0, color="r", label="val")
        plt.title("CNN sentiment")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.show()

    def predict(self, data):
        return self.model.predict(data)
