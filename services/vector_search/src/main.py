from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

from src.serve_onnx import onnx_vectorizer

# Initialize FastAPI app
app = FastAPI()

# Load the model and move it to GPU if available
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Pydantic model to define the input structure
class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def get_embeddings(input_data: TextInput):
    try:
        # Tokenize the input text
        sentences = [input_data.text]
        embeddings = model.encode(sentences)
        return {"embedding": embeddings[0].tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the service, use the following command:
# uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
