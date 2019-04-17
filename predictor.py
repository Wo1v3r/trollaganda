class Predictor(object):
    def __init__(self, model = None, embed = None):
        self.model = model
        self.embed = embed

    # TODO: Create a Tokenizer Class for PrepareEmbedding and Predictor to share [Maybe Use Preprocess class]

    def predict(self, messages):
        data = self.embed.preprocess_predictions(messages)
        predictions = self.model.predict(data)

        results = []

        for i in range(len(messages)):
            isTroll = predictions[i] > 0.5
            print("Message: [%s], Prediction: [%s]" % (
            messages[i], "Troll" if isTroll else "Not a Troll"))
            results.append(isTroll)

        return results
