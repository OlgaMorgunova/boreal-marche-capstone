# Architecture

## Offline training / preparation

```text
UCI Online Retail II
        |
        v
  Reproducible ETL
        |
        +--------------------+
        |                    |
        v                    v
 Retention features     Basket transactions
        |                    |
        v                    v
 Head A models          Head B rules
        |                    |
        +---------+----------+
                  |
                  v
        MLflow / artifacts
                  |
                  +----> RAG corpus + retrieval index
```

## Online serving

```text
Frontend (Streamlit)
        |
        v
     FastAPI
   / /predict /recommend /chat
        |
   +----+------------------+
   |       |       |       |
   v       v       v       v
Head A   Rules   RAG     Policy/Product tools
   |       |       |       |
   +-------+-------+-------+
           |
     structured logs
           |
           v
      Monitoring
```

The final architecture will be updated as implementation decisions are validated.
