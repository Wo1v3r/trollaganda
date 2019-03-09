##### Installation:
venv/bin/pip install -r venv/requirements.txt

##### Authenticate Kaggle:
- Go to Kaggle > My Account > Create new API Token
- Copy kaggle.json to ~/.kaggle/kaggle.json (unix)

##### Download Dataset:
kaggle datasets download -d liranr23/troll-or-not
kaggle datasets download -d umbertogriffo/googles-trained-word2vec-model-in-python

#### After adding packages make sure you:
venv/bin/pip freeze > requirements.txt