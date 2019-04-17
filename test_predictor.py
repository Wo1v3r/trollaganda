from predictor import Predictor

import unittest



class MockEmbed(object):
    def preprocess_predictions(self, messages):
        return map(lambda msg : "preprocessed " + msg, messages)

class MockModel(object):
    def predict(self, data):
        return list(map(lambda d: "troll" in d and "preprocessed" in d, data))

class TestPredictorMethods(unittest.TestCase):

    def test_init(self):
        messages = [
            "troll message",
            "elf message",
            "message elf",
            "message troll",
        ]

        embed = MockEmbed()
        model = MockModel()

        predictor = Predictor(embed=embed, model=model)

        predictions = predictor.predict(messages=messages)

        self.assertEqual(predictions, [True, False, False, True])
if __name__ == '__main__':
    unittest.main()