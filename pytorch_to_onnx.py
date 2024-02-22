import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

# Load the fine-tuned model
path_to_model = "model2"
model = AutoModelForSequenceClassification.from_pretrained(path_to_model)

# Example input text
input_text = "Indian Street"

tokenized_input =tokenizer(input_text, padding=True, truncation=True, max_length= 28, return_tensors="pt")

# Make predictions using the PyTorch model
with torch.no_grad():
    outputs = model(**tokenized_input)
logits = outputs.logits
predicted_labels = logits.argmax(dim=1)

# Export the PyTorch model to ONNX
batch_size = 1
x = torch.randint(0, 10000, (batch_size, 28), dtype=torch.long)  # Example token IDs, adjust as needed
y = torch.randint(0, 10000, (batch_size, 28), dtype=torch.long)  # Example token IDs, adjust as needed

torch.onnx.export(model,
                  (x,y),
                  "model2.onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size'},
                                'attention_mask': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})


print("Model exported to 'model.onnx'")
