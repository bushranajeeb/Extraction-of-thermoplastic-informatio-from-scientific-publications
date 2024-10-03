import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load your annotated data (replace 'dataset.jsonl' with your actual dataset path)
with open("thermoplastic_ner_annotated.jsonl", "r") as f:
    raw_data = [json.loads(line) for line in f]

# Prepare data for Hugging Face's Dataset format
def prepare_data(data):
    examples = []
    for entry in data:
        tokens = [token['text'] for token in entry['tokens']]
        labels = ['O'] * len(tokens)  # Initialize with 'O' for tokens not labeled

        for span in entry['spans']:
            # Assign label to corresponding token spans
            for i in range(span['token_start'], span['token_end'] + 1):
                labels[i] = span['label']

        examples.append({
            'tokens': tokens,
            'ner_tags': labels,
        })
    return examples

prepared_data = prepare_data(raw_data)

# Create a Dataset from the prepared data
dataset = Dataset.from_list(prepared_data)

# View the dataset
print(dataset)

# Define your NER labels (including 'O' for non-entity tokens)
label_list = ['O', 'MATERIAL', 'MATERIAL_PROPERTY', 'PROP_VALUE', 'AUTOMOTIVE_PART']

# Create a mapping from label names to integers
label_to_id = {label: i for i, label in enumerate(label_list)}

# Print the mapping to ensure correctness
print("Label to ID Mapping: ", label_to_id)

# Function to map string labels to integers
def label_to_id_mapper(examples):
    examples['ner_tags'] = [label_to_id[label] for label in examples['ner_tags']]
    return examples

# Apply the label mapping to the dataset
mapped_dataset = dataset.map(label_to_id_mapper)

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize the inputs and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their original words
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]  # Assign -100 to padding
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Tokenize the dataset
tokenized_dataset = mapped_dataset.map(tokenize_and_align_labels, batched=True)
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# This will create a dataset with 'train' and 'test' splits
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Load pre-trained model for token classification (you can replace 'bert-base-uncased' with another model)
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list)
)

# Data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Output directory
    evaluation_strategy="epoch",      # Evaluate every epoch
    save_strategy="epoch",            # Save checkpoint every epoch
    logging_dir="./logs",             # Logging directory
    per_device_train_batch_size=8,    # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=5,               # Number of training epochs
    learning_rate=5e-5,               # Learning rate
    weight_decay=0.01,                # Weight decay
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
)

# Get the label list and the id-to-label mapping
label_list = ['O', 'MATERIAL', 'MATERIAL_PROPERTY', 'PROP_VALUE', 'AUTOMOTIVE_PART']
id_to_label = {i: label for i, label in enumerate(label_list)}

# Define a function to align predictions and true labels (same as before)
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)

    # Convert the predicted and true labels to list of strings
    true_labels = [[id_to_label[l] for l in label_id if l != -100] for label_id in label_ids]
    pred_labels = [[id_to_label[p] for p, l in zip(pred, label_id) if l != -100] for pred, label_id in zip(preds, label_ids)]

    return pred_labels, true_labels

# Compute metrics function
def compute_metrics(p):
    pred_labels, true_labels = align_predictions(p.predictions, p.label_ids)

    # Flatten the labels for calculating metrics
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    pred_labels_flat = [item for sublist in pred_labels for item in sublist]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels_flat, pred_labels_flat)

    # Calculate precision, recall, f1 (optional)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, pred_labels_flat, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Trainer setup with the custom compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the split train dataset
    eval_dataset=eval_dataset,    # Use the split eval dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Pass the compute_metrics function here
)

# Start training
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation Results: {eval_results}")

# Save the trained model and tokenizer
model.save_pretrained("/app/output_model/fine_tuned_ner_model")
tokenizer.save_pretrained("/app/output_model/fine_tuned_ner_model")

