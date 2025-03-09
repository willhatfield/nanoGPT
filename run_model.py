import torch
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig

# Define the configuration for the GPT model
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)

# Initialize the model
model = GPT(config)

# Load the model onto the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Interact with the user to get input text
sample_text = input("Enter a prompt: ")

# Preprocess the input text to get token indices
sample_input = tokenizer.encode(sample_text, return_tensors="pt").to(device)

# Generate text using the model
with torch.no_grad():
    output = model.generate(sample_input, max_new_tokens=50, temperature=1.0, top_k=10)

# Decode the generated token indices back to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated output
print("Generated output:", generated_text)