import io
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from PIL import Image

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import MultimodalSearch
from src.pipline.training_pipeline import TrainingPipeline
from src.exception import MyException
from src.logger import logger


# -------------------------------
# App Initialization
# -------------------------------

app=FastAPI()
app.mount("/static",StaticFiles(directory="static"),name="static")
templates=Jinja2Templates(directory="templates")



# -------------------------------
# CORS (Allow frontend JS)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Home Page
# -------------------------------
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
        request=request,
        name="mltimodal.html"
    )

# -------------------------------
# TRAIN PIPELINE
# -------------------------------
@app.get("/train", tags=["Training"])
async def train():
    try:
        logger.info("Starting training pipeline...")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        return Response("Training successful!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return Response(f"Error: {e}")


# -------------------------------
# TEXT SEARCH (FORM)
# -------------------------------
@app.post("/", tags=["Prediction"])
async def predict_text(request: Request):
    try:
        form = await request.form()
        query = form.get("query")

        if not query:
            return templates.TemplateResponse(
                request=request,
                name="mltimodal.html",
                context={"results": [], "error": "Empty query"}
            )

        logger.info(f"Text query received: {query}")

        model = MultimodalSearch()
        results = model.predict(query)

        return templates.TemplateResponse(
            request=request,
            name="mltimodal.html",
            context={"results": results}
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return templates.TemplateResponse(
            request=request,
            name="mltimodal.html",
            context={"error": str(e)}
        )


# -----------------------------
# IMAGE SEARCH (FORM / API)
# -------------------------------
@app.post("/image", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Image received: {file.filename}")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = MultimodalSearch()
        results = model.predict(image)

        return JSONResponse({"results": results})

    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        return JSONResponse({"error": str(e)})


# -------------------------------
# TEXT SEARCH API (for JS frontend)
# -------------------------------
@app.post("/api/search/text", tags=["API"])
async def api_text_search(query: str = Form(...)):
    try:
        model = MultimodalSearch()
        results = model.predict(query)

        return {"results": results}

    except Exception as e:
        logger.error(f"Text search error: {e}")
        return {"error": str(e), "message": "Please train the model first by visiting /train endpoint if model files are missing."}


# -------------------------------
# IMAGE SEARCH API (for JS frontend)
# -------------------------------
@app.post("/api/search/image", tags=["API"])
async def api_image_search(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = MultimodalSearch()
        results = model.predict(image)

        return {"results": results}

    except Exception as e:
        logger.error(f"Image search error: {e}")
        return {"error": str(e), "message": "Please train the model first by visiting /train endpoint if model files are missing."}


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "_main_":
    app_run(app, host=APP_HOST, port=APP_PORT)