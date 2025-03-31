# generate_predictions.py
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
import os

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('/lustre/work/cijezue/Hope/bert_tokenizer')

# Load models
binary_model = tf.keras.models.load_model('/lustre/work/cijezue/Hope/bert_binary_model')
multi_model = tf.keras.models.load_model('/lustre/work/cijezue/Hope/bert_multi_model')

# Load test data (assuming en.csv has a 'text' column)
test_df = pd.read_csv('en_test_without_labels.csv')
texts = test_df['text'].tolist()

# Tokenize texts
def encode_texts(texts, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='tf')

encodings = encode_texts(texts)

# Prepare inputs as a dictionary
inputs = {
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'token_type_ids': encodings.get('token_type_ids', None)
}

# Reverse label mappings
binary_label_map = {0: 'Not Hope', 1: 'Hope'}
multi_label_map = {
    0: 'Not Hope',
    1: 'Generalized Hope',
    2: 'Realistic Hope',
    3: 'Unrealistic Hope',
    4: 'Sarcasm'
}

# Predict for Subtask 1 (Binary)
binary_pred = binary_model(inputs, training=False)
binary_labels = tf.argmax(binary_pred['logits'], axis=1).numpy()
binary_tags = [binary_label_map[label] for label in binary_labels]

# Predict for Subtask 2 (Multiclass)
multi_pred = multi_model(inputs, training=False)
multi_labels = tf.argmax(multi_pred['logits'], axis=1).numpy()
multi_tags = [multi_label_map[label] for label in multi_labels]

# Create DataFrames
binary_df = pd.DataFrame({'Text': texts, 'Tag': binary_tags})
multi_df = pd.DataFrame({'Text': texts, 'Tag': multi_tags})

# Save predictions.csv for each subtask
binary_df.to_csv('/lustre/work/cijezue/Hope/predictions_binary.csv', index=False)
multi_df.to_csv('/lustre/work/cijezue/Hope/predictions_multi.csv', index=False)

print("Predictions saved to predictions_binary.csv and predictions_multi.csv")