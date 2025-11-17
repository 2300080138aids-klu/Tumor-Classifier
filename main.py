from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_mri
import tempfile
import os

app = FastAPI(title="MRI Tumor Classifier API")

# =====================================================
# ✅ CORS Settings for React (Vite frontend)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ✅ Root Endpoint
# =====================================================
@app.get("/")
async def root():
    return {"message": "✅ MRI Tumor Classifier backend is running."}

# =====================================================
# ✅ Prediction Endpoint
# =====================================================
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        # Run prediction
        result = predict_mri(temp_path)
    except Exception as e:
        result = {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result
