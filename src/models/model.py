import torch
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)



# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 32

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 60

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
model_name_or_path = "gpt2"

# Dictionary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {"neg": 0, "pos": 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

# Get model configuration.
print("Loading configuraiton...")
model_config = GPT2Config.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels
)

# Get model's tokenizer.
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path
)
# default to left padding
tokenizer.padding_side = "left"   #NB: Needs better understanding
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token  #NB: Needs better understanding


# Get the actual model.
print("Loading model...")
model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, config=model_config
)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print("Model loaded to `%s`" % device)
