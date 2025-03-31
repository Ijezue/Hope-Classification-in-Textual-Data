# hope_classifier.py
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

# Load data
train_df = pd.read_csv('en_train.csv')
test_df = pd.read_csv('en_dev.csv')

# BERT Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def encode_texts(texts, max_length=128):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='tf')

# Split training data into train and validation sets
train_texts, val_texts, train_binary_labels, val_binary_labels = train_test_split(
    train_df['text'], train_df['binary'], test_size=0.2, random_state=42
)
train_multi_labels, val_multi_labels = train_test_split(
    train_df['multiclass'], test_size=0.2, random_state=42
)

train_encodings = encode_texts(train_texts)
val_encodings = encode_texts(val_texts)
test_encodings = encode_texts(test_df['text'])

# Label mapping
binary_label_map = {'Not Hope': 0, 'Hope': 1}
multi_label_map = {'Not Hope': 0, 'Generalized Hope': 1, 'Realistic Hope': 2, 'Unrealistic Hope': 3, 'Sarcasm': 4}

y_train_binary = train_binary_labels.map(binary_label_map)
y_val_binary = val_binary_labels.map(binary_label_map)
y_test_binary = test_df['binary'].map(binary_label_map)
y_train_multi = train_multi_labels.map(multi_label_map)
y_val_multi = val_multi_labels.map(multi_label_map)
y_test_multi = test_df['multiclass'].map(multi_label_map)

# Binary Model
model_binary = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model_binary.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callback to save the best model
checkpoint_binary = tf.keras.callbacks.ModelCheckpoint(
    '/lustre/work/cijezue/Hope/bert_binary_model',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_format='tf'
)

model_binary.fit(
    [train_encodings['input_ids'], train_encodings['attention_mask']],
    y_train_binary,
    validation_data=([val_encodings['input_ids'], val_encodings['attention_mask']], y_val_binary),
    epochs=3,  
    batch_size=8,
    callbacks=[checkpoint_binary],
    verbose=1
)

# Evaluate on test set
binary_pred = model_binary.predict([test_encodings['input_ids'], test_encodings['attention_mask']])
binary_pred_labels = tf.argmax(binary_pred.logits, axis=1)
binary_acc = accuracy_score(y_test_binary, binary_pred_labels)

# Multiclass Model
model_multi = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model_multi.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callback to save the best model
checkpoint_multi = tf.keras.callbacks.ModelCheckpoint(
    '/lustre/work/cijezue/Hope/bert_multi_model',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_format='tf'
)

model_multi.fit(
    [train_encodings['input_ids'], train_encodings['attention_mask']],
    y_train_multi,
    validation_data=([val_encodings['input_ids'], val_encodings['attention_mask']], y_val_multi),
    epochs=3,  
    batch_size=8,
    callbacks=[checkpoint_multi],
    verbose=1
)

# Evaluate on test set
multi_pred = model_multi.predict([test_encodings['input_ids'], test_encodings['attention_mask']])
multi_pred_labels = tf.argmax(multi_pred.logits, axis=1)
multi_acc = accuracy_score(y_test_multi, multi_pred_labels)

# Save tokenizer
tokenizer.save_pretrained('/lustre/work/cijezue/Hope/bert_tokenizer')

# Save results
with open('/lustre/work/cijezue/Hope/out/hope_results.txt', 'w') as f:
    f.write(f"Binary Accuracy: {binary_acc}\n")
    f.write(f"Multiclass Accuracy: {multi_acc}\n")

print(f"Binary Accuracy: {binary_acc}")
print(f"Multiclass Accuracy: {multi_acc}")
print("Best models saved to /lustre/work/cijezue/Hope/bert_binary_model and /lustre/work/cijezue/Hope/bert_multi_model")
