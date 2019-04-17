class Predictor(object):
    def __init__(self, model = None, embed = None):
        self.model = model
        self.embed = embed

    # TODO: Create a Tokenizer Class for PrepareEmbedding and Predictor to share [Maybe Use Preprocess class]

    def predict(self, messages):
        data = self.embed.preprocess_predictions(messages)
        predictions = self.model.predict(data)

        for i in range(len(messages)):
            print("Message: [%s], Prediction: [%s]" % (
            messages[i], "Troll" if predictions[i] > 0.5 else "Not a Troll"))

