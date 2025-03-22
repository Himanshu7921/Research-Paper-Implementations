import pandas as pd
import torch.nn.functional as F
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import torch
import torchmetrics.classification
import torch.nn as nn
import pickle


class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features = 1):
        super().__init__()

        self.model = nn.Sequential(
          nn.Linear(in_features, out_features) # Applies a transformation to the provided vector filed as y = x * W.T + b
        )

    def forward(self, x: torch.tensor):
        return torch.sigmoid(self.model(x))

def load_data():
    # Data Preprocessing
    PATH = "D:\Code Playground\Solving Research Paper\Data_sets\email_spam_dataset.csv"
    data = pd.read_csv(PATH, encoding='latin-1')

    data = data.rename(columns={"v1": "Labels", "v2": "Email"})
    data['Labels'] = data['Labels'].apply(lambda x: 0 if x == "ham" else 1)

    # Preprocessing emails
    stemmer = PorterStemmer()
    corpus = []
    nltk.download('stopwords')
    sp_words = set(stopwords.words('english'))

    for i in range(len(data)):
        emails = data['Email'].iloc[i].lower()
        emails = emails.translate(str.maketrans('', '', string.punctuation)).split()
        emails = [stemmer.stem(word) for word in emails if word not in sp_words]
        emails = ' '.join(emails)
        corpus.append(emails)

    # Vectorization
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    Y = data['Labels']

    with open(r"D:\Code Playground\Solving Research Paper\Logistic Regression\vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    return torch.tensor(X).type(torch.float32), torch.tensor(Y).type(torch.float32)

class Train_Test_Model():
    def __init__(self, X: torch.tensor, y:torch.tensor, model: LogisticRegression, epochs: int, loss_fn: torch.nn, optimizer: torch.optim):
        self.X = X
        self.y = y
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_function = loss_fn
        self.__get_train_test_split()
    
    def __get_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size = 0.3)
        
    
    def train(self):
        print("Training Model.....")
        self.model.train()
        for epoch in range(self.epochs):
            # Get the Model predictions
            y_preds = self.model(self.X_train).squeeze()
            # Calculate the loss
            loss = self.loss_function(y_preds, self.y_train)
            # Perform Descent gradient and update the model's learnable paraameters
            self.optimizer.zero_grad()
            # Calculate the loss backward
            loss.backward()
            # optimizer step
            self.optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch [{epoch} / {self.epochs}] | loss: {loss:.4f}")
        print("Training Completed ✅")
        return self.model
    
    def test(self):
        print(f"Testing Model....")
        y_preds_test = []
        accuracy = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
        with torch.inference_mode():
            y_preds = self.model(self.X_test).squeeze()
            y_preds_test.append(y_preds)
            loss = self.loss_function(y_preds, self.y_test)
            acc = accuracy(y_preds, self.y_test)
            print(f"loss: {loss:.2f} | Accuracy: {acc:.4f}")

        print("Testing Completed ✅")

        with open(r"D:\Code Playground\Solving Research Paper\Logistic Regression\model.pkl", "wb") as model_file:
            pickle.dump(self.model, model_file)
        return y_preds_test
    
    @staticmethod
    def predict(model, X):
        model.eval()
        with torch.inference_mode():
            y_pred = model(X)
            binary_preds = (y_pred >= 0.5).float()  # Converts probabilities to 0s and 1s
        return binary_preds

def main():
    # Load daatset
    X,y = load_data()
    # load Model
    model = LogisticRegression(X.shape[1], 1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Load Model_trainer_tester
    model_trainer = Train_Test_Model(X, y, model, 10000, loss_fn, optimizer)
    # train model
    trained_model = model_trainer.train()
    # test model
    test_results = model_trainer.test()
    # Save trained model
    torch.save(trained_model.state_dict(), r"D:\Code Playground\Solving Research Paper\Logistic Regression\spam_classifier.pkl")
    print("Model saved successfully!")
    
if __name__ == "__main__":
    main()