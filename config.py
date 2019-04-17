import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stopwords0 = set(stopwords.words('english'))
SPLITDATA = 10000
EMBEDDINGDIM = 300
MAXVOCABSIZE = 175303
MAXSEQLENGTH = 200
BATCHSIZE = 256
EPOCHS = 3
LOAD_MODEL_PATH = "./model"
