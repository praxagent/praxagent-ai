import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import torch.nn.functional as F
import nltk

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

doc = """
Our machine learning infrastructure consists of multiple processing pipelines that handle various data transformation tasks. The critical production system for customer data processing generates comprehensive analytics reports and specific outputs including detailed performance metrics for the marketing team. This system runs batch jobs every hour and outputs detailed performance metrics, user engagement statistics, and conversion rate analysis with specific outputs tailored for customer support optimization.

The secondary critical production pipeline focuses on financial data validation and compliance checking with specific outputs for customer risk assessment. It processes transaction records, validates merchant information, and generates audit trail documentation for regulatory purposes. The system also produces compliance reports, risk assessment matrices, and fraud detection alerts that are sent to the security team with specific outputs designed for customer protection services.

Our experimental AI research division has been working on several prototype models for natural language processing tasks. These models generate synthetic training data, produce text embeddings for similarity search, and create automated content summaries for internal documentation. The research team uses these outputs to benchmark against industry standards.

The production deployment infrastructure uses containerized microservices to handle real-time data processing. Each service generates its own telemetry data, produces health check responses, and outputs structured logging information for monitoring purposes. The main application generates user session tokens, processes authentication requests, and produces encrypted data exports.

In the manufacturing sector integration, our critical production system for customer equipment monitoring processes sensor data from factory equipment. The system generates predictive maintenance schedules, produces equipment efficiency reports, and outputs quality control metrics with specific outputs for customer satisfaction tracking. Additionally, it generates automated alerts for equipment failures and produces calibration recommendations for precision instruments with specific outputs tailored for customer support teams.

The data warehouse ETL processes handle large-scale data transformation tasks across multiple business units. These critical production systems generate data quality reports, produce schema validation results, and output data lineage documentation with specific outputs for customer data protection. The warehouse system also generates automated backup schedules and produces disaster recovery checkpoint files designed for customer service continuity.

Our customer-facing application backend processes user requests and generates personalized recommendation engines with critical production system capabilities. The recommendation system produces user preference profiles, generates product similarity scores, and outputs dynamic pricing suggestions. The core application stack generates session management tokens and produces real-time user activity feeds with specific outputs for customer support analytics.

The main ticketing platform that we deployed last quarter for handling high-priority client requests processes incoming help desk inquiries and creates automated response templates. This platform specifically produces intelligent reply suggestions, creates escalation priority scores, and delivers satisfaction prediction metrics that help the help desk team prioritize their workload efficiently.
"""

query = "What specific outputs does the critical production system generate for customer support?"

def embed(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Traditional chunking
chunks = sent_tokenize(doc)
traditional = [embed(c) for c in chunks]
query_vec = embed(query)

print("Traditional Chunking (Top 5 most similar):")
# Calculate similarities for all chunks
similarities = []
for i, vec in enumerate(traditional):
    sim = F.cosine_similarity(query_vec, vec, dim=0).item()
    similarities.append((sim, i, chunks[i]))

# Sort by similarity (descending) and take top 5
top_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]

# Display top 5
for rank, (sim, chunk_idx, chunk_text) in enumerate(top_similarities, 1):
    print(f"Rank {rank} - Chunk {chunk_idx+1}: {chunk_text}")
    print(f"Similarity: {sim:.4f}\n")
