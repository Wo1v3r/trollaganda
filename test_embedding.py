from embedding import PrepareEmbedding, PreProcess
import pandas as pd
import unittest


# class PreProcess(object):
#     """
#     This class will pre-process pandas data frame
#     """
#     def __init__(self, data, textfield):
#         self.data = data
#         self.textfield = textfield
#
#     def process_text(self):
#         self.data[self.textfield] = self.data[self.textfield].str.replace(r"http\S+", "LINK")
#         self.data[self.textfield] = self.data[self.textfield].str.replace(r"@\S+", "TAG")
#         self.data[self.textfield] = self.data[self.textfield].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
#         self.data[self.textfield] = self.data[self.textfield].str.replace(r"@", "AT")
#         self.data[self.textfield] = self.data[self.textfield].str.lower()
#         return self.data
#
#     def see_data_head(self):
#         self.data.head()
#


class TestEmbeddingMethods(unittest.TestCase):

    def test_init(self):
        X = ["one","two","three","four"]
        Y = [1,2,2,1]
        embedded_path = 'embedded_path'

        embed = PrepareEmbedding(X=X, Y=Y, embedded_path=embedded_path, test_size=0.5)

        self.assertEqual(embed.embedded_path, embedded_path, "Sets path to embedding")
        self.assertEqual(len(embed.X_test), len(X) * 0.5, "Splits test data to passed size")
        self.assertLessEqual(embed.Y_test[0], 1, "Transforms labels <= 1")
        self.assertGreaterEqual(embed.Y_test[0], -1, "Transforms labels >= -1")


class TestPreprocessMethods(unittest.TestCase):

    def test_init(self):
        data = 'data'
        textfield = 'textfield'

        preprocess = PreProcess(data = data, textfield= textfield)

        self.assertIs(preprocess.data, data)
        self.assertIs(preprocess.textfield, textfield)

    def test_process_text(self):
        data = pd.DataFrame({"message": ["https://google.com"]})
        textfield = "message"

        preprocess = PreProcess(data =data, textfield=textfield)

        processed_text = preprocess.process_text()

        self.assertEqual(processed_text.message[0], "link")


if __name__ == '__main__':
    unittest.main()