import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

doc = "The DataProcessor class is initialized with a configuration object. It has a method called process that takes raw data and cleans it. This method returns a cleaned DataFrame."
query = "What does the process method return?"

def embed(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Traditional chunking
chunks = sent_tokenize(doc)
traditional = [embed(c) for c in chunks]
query_vec = embed(query)

print("Traditional Chunking:")
for i, vec in enumerate(traditional):
    sim = F.cosine_similarity(query_vec, vec, dim=0).item()
    print(f"Chunk {i+1}: {chunks[i]}")
    print(f"Similarity: {sim:.4f}\n")
