import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from utils import save_as_json, open_json, set_workspace, Logger
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel
import random
import time
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=4)
parser.add_argument('-lr', '--learning_rate', type=int, default=0)
args = parser.parse_args()

learning_rates = [5e-5, 3e-5, 2e-5]
EPOCHS = args.epochs
LR = learning_rates[args.learning_rate]

set_workspace()


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('./models/chinese-bert-wwm-ext', local_files_only=True)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[:]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# setup and load data


data = pd.DataFrame(open_json('./data/sentiment/train/usual_train.json'))
data_ = pd.DataFrame(open_json('./data/sentiment/train/virus_train.json'))
data = data.append(data_)

# data = data[:64]

label_mapping = {
    'angry':0,
    'happy':1,
    'neutral':2,
    'surprise':3,
    'fear':4,
    'sad':5
}
data['label'] = data['label'].map(label_mapping)


X = data.content.values
y = data.label.values

X_train, X_val, y_train, y_val =\
    train_test_split(X, y, test_size=0.1, random_state=2020)

# Load test data
# test_data = pd.DataFrame(open_json('data/sentiment/eval（刷榜数据集）/usual_eval.txt'))
# test_data_ = pd.DataFrame(open_json('data/sentiment/eval（刷榜数据集）/virus_eval.txt'))
# test_data.append(test_data_)

# test_data = pd.DataFrame(open_json('data/cleaned_data/weibo_contents_no_tag.json'))
# inference with all weibos with origin
# test_data = pd.DataFrame(open_json('data/cleaned_data/weibos_content_with_origin.json'))
test_data = pd.DataFrame(open_json('data/cleaned_data/weibo_contents_other.json'))
# Keep important columns
# test_data = test_data[['id', 'content']]
test_data = test_data[['weibo_id', 'content']]

# Display 5 samples from the test data
test_data.sample(5)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Concatenate train data and test data
all_contents = np.concatenate([data.content.values, test_data.content.values])

# Encode our concatenated data
encoded_contents = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_contents]

# Find the maximum length
max_len = max(len(sent) for sent in encoded_contents)
print('Max length: ', max_len)

# Specify `MAX_LEN`
MAX_LEN = 256

# Print sentence 0 and its encoded token ids
token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
print('Original: ', X[0])
print('Token IDs: ', token_ids)

# Run function `preprocessing_for_bert` on the train set and the validation set
print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
batch_size = 32

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Train model
# Create the BertClassfier class


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 6

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('./models/chinese-bert-wwm-ext', local_files_only=True)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        return self.classifier(last_hidden_state_cls)


def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# Specify loss function
loss_fn = nn.CrossEntropyLoss()


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, optimizer, scheduler, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


# set_seed(42)    # Set seed for reproducibility
# bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)
# train(bert_classifier, train_dataloader, val_dataloader, epochs=EPOCHS, evaluation=True)


# eval
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc


def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), len(num_classes))
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """

    # preds = probs[:, 1]

    # fpr, tpr, threshold = roc_curve(y_true, preds)
    # roc_auc = auc(fpr, tpr)
    roc_auc_score = (y_true, probs)
    # print(f'AUC: {roc_auc_score:.4f}')
    print(f'AUC: {roc_auc_score}')

    # Get accuracy over the test set
    # y_pred = np.where(preds >= 0.5, 1, 0)
    y_pred = np.argmax(probs, axis=1)
    # accuracy = accuracy_score(y_true, y_pred)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Accuracy: {accuracy * 100}%')


def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    return F.softmax(all_logits, dim=1).cpu().numpy()


# Train the Bert Classifier on the entire training data
set_seed(42)
# bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)
# train(bert_classifier, full_train_dataloader, epochs=EPOCHS)

print('Saving and loading model...')
model_path = './models/chinese-bert-wwm-ext/fine_tuned_model'
# torch.save(bert_classifier.state_dict(), model_path)
# torch.save(bert_classifier, './models/')
#
# saved_model = torch.load('./models/chinese-bert-wwm-ext/')

# load and predict
# bert_classifier

bert_classifier = BertClassifier(freeze_bert=False)
bert_classifier.to(device)
bert_classifier.load_state_dict(torch.load(model_path))
# Tell PyTorch to run the model on GPU
print('model loaded.')
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(test_data.content)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

# Compute predicted probabilities on the test set
print('Predicting....')
probs = bert_predict(bert_classifier, test_dataloader)
probs_with_id = list(zip(test_data.weibo_id, probs.tolist()))
save_as_json(probs_with_id, './data/results/sentiment_predict_0312.json')
print('Prediction Saved.')


