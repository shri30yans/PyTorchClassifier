import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self, text_features, numerical_features):
        self.text_features = text_features
        self.numerical_features = numerical_features
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def preprocess(self, dataframe):
        for feature in self.text_features:
            dataframe[feature] = self.encoder.fit_transform(dataframe[feature])

        dataframe[self.numerical_features] = self.scaler.fit_transform(dataframe[self.numerical_features])
        return dataframe

class ModelTrainer:
    def __init__(self, model, loss_function, optimizer, num_epochs):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y.view(-1, 1))
                loss.backward()
                self.optimizer.step()

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, val_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                predicted = (outputs >= 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y.view(-1, 1)).sum().item()
        accuracy = 100 * (correct / total)
        return accuracy

def load_data(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe

def prepare_data(dataframe, text_features, numerical_features):
    preprocessor = DataPreprocessor(text_features, numerical_features)
    encoded_dataframe = preprocessor.preprocess(dataframe)
    features = encoded_dataframe.drop(columns=["LeaveOrNot"])
    target = encoded_dataframe["LeaveOrNot"]
    return features, target

def create_datasets(features, target, train_size):
    X_train = features[:train_size]
    y_train = target[:train_size]
    X_val = features[train_size:]
    y_val = target[train_size:]

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    return train_dataset, val_dataset

def main():
    # Configuration
    file_path = "dataset.csv"
    text_features = ["Education", "City", "Gender", "EverBenched"]
    numerical_features = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]
    train_size = 7  # 80% of the data for training

    # Load data
    dataframe = load_data(file_path)

    # Prepare data
    features, target = prepare_data(dataframe, text_features, numerical_features)

    # Create datasets
    train_dataset, val_dataset = create_datasets(features, target, train_size)

    # Define model
    model = nn.Sequential(
        nn.Linear(len(text_features) + len(numerical_features), 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # Define loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train model
    trainer = ModelTrainer(model, loss_function, optimizer, num_epochs=100)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    trainer.train(train_loader)

    # Evaluate model
    evaluator = ModelEvaluator(model)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    accuracy = evaluator.evaluate(val_loader)

    print(f"Accuracy on the validation set: {accuracy}%")

if __name__ == "__main__":
    main()
