from embedding import PrepareEmbedding

import unittest

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


if __name__ == '__main__':
    unittest.main()