```
graph TD
    A["Original Document"] --> B["Traditional Chunking"]
    A --> C["Late Chunking"]
    
    B --> D["Fixed Chunks<br/>(150 words)"]
    D --> E["Independent<br/>Embeddings"]
    E --> F["Cosine Similarity<br/>Search"]
    
    C --> G["Context Windows<br/>(400 words)"]
    G --> H["Context-Aware<br/>Embeddings"]
    H --> I["Return Focused<br/>Chunks (150 words)"]
    I --> J["Cosine Similarity<br/>Search"]
    
    K["Query:<br/>'What does it return?'"] --> F
    K --> J
    
    F --> L["❌ Lost Pronoun<br/>Reference"]
    J --> M["✅ Preserved<br/>Context"]
    
    style L fill:#ffebee
    style M fill:#e8f5e8
    style A fill:#e3f2fd
    style K fill:#fff3e0
```