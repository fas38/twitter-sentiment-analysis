# Imports
import pandas as pd
import numpy as np
import os
import warnings
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModel
from transformers import TrainerCallback, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import gc
import optuna
from GPUtil import showUtilization as gpu_usage
import pickle

# Load data

team_id = '20' #put your team id here
split = 'test_1' # replace by 'test_2' for FINAL submission
df = pd.read_csv('dataset/tweets_train.csv')
df_test = pd.read_csv(f'dataset/tweets_{split}.csv')
df['words_str'] = df['words'].apply(lambda words: ' '.join(eval(words)))
df_test['words_str'] = df_test['words'].apply(lambda words: ' '.join(eval(words)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# clear gpu memory
gc.collect()
torch.cuda.empty_cache()
# gpu_usage()

X = df['words_str']
y_text = df['sentiment']
# y_text = df.sentiment.values
le = preprocessing.LabelEncoder()
le.fit(y_text)
print(f'Original classes {le.classes_}')
print(f'Corresponding numeric classes {le.transform(le.classes_)}')
y =le.transform(y_text)
print(f"X: {X.shape}")
print(f"y: {y.shape} {np.unique(y)}")

# Splitting
train_texts, val_texts, train_labels, val_labels = train_test_split(df['words_str'], y, test_size=0.2, shuffle=True, random_state=42)

# Tokenize the input
tokenizer_twitter = AutoTokenizer.from_pretrained('models/twitter-roberta-base/')
tokenizer = tokenizer_twitter
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)


class ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.astype('int') # Change to integer type

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Change to long type for classification
        return item

    def __len__(self):
        return len(self.labels)
    
# Function to compute f1_macro
def f1_macro(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {'f1_macro': f1_score(labels, predictions, average='macro')}

class RobertaClassificationTwitter_2(nn.Module):
    def __init__(self, dropout=0.1):
        super(RobertaClassificationTwitter_2, self).__init__()
        # self.roberta = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
        self.roberta = AutoModel.from_pretrained('models/twitter-roberta-base-sentiment-latest/')
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.roberta.config.hidden_size

        # Adding an additional hidden layer
        self.hidden_layer = nn.Linear(hidden_size, hidden_size//2)
        
        # Adding L2 regularization (weight decay) to the hidden layer
        self.regularization = nn.LayerNorm(hidden_size//2)
        
        # Final classification layer with 3 classes
        self.classifier = nn.Linear(hidden_size//2, 3)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        # Passing through the hidden layer with ReLU activation
        hidden_output = self.hidden_layer(pooled_output)
        hidden_output = F.relu(hidden_output)
        
        # Applying Layer Normalization (regularization)
        hidden_output = self.regularization(hidden_output)
        
        logits = self.classifier(hidden_output)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# dataset 
train_dataset = ClassificationDataset(train_encodings, train_labels)
val_dataset = ClassificationDataset(val_encodings, val_labels)

# hyperparameter tuning with optuna
warnings.filterwarnings('ignore')
epochs = 40
num_trials = 30
def objective(trial):
    # hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = RobertaClassificationTwitter_2(dropout=dropout).to(device)
    
    # Training arguments with hyperparameters
    training_args = TrainingArguments(
        output_dir='./output',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_dir='./logs',
        evaluation_strategy='epoch',
        # logging_steps=100, # Set to evaluate every 100 steps
        logging_strategy='epoch',
        weight_decay=weight_decay,
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
    )

    # Model & Trainer
    model = RobertaClassificationTwitter_2(dropout=dropout).to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=f1_macro,
    )

    # Train and evaluate the model
    trainer.train()
    eval_results = trainer.evaluate()
    
    # Return the evaluation metric
    return eval_results["eval_f1_macro"]

# trial study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=num_trials) 

# Results
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
