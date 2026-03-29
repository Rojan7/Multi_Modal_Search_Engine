import io
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.logger import logger


app = FastAPI()

Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_search_model():
    from src.pipline.prediction_pipeline import MultimodalSearch

    return MultimodalSearch()


def load_training_pipeline():
    from src.pipline.training_pipeline import TrainingPipeline

    return TrainingPipeline()


@app.get("/", tags=["UI"])
async def index(request: Request):
    return templates.TemplateResponse("mltimodal.html", {"request": request})


@app.get("/train", tags=["Training"])
async def train():
    try:
        logger.info("Starting training pipeline...")
        train_pipeline = load_training_pipeline()

        if hasattr(train_pipeline, "run_pipeline"):
            train_pipeline.run_pipeline()
        else:
            train_pipeline.run_pipline()

        return Response("Training successful!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return Response(f"Error: {e}")


@app.post("/", tags=["Prediction"])
async def predict_text(request: Request):
    try:
        form = await request.form()
        query = form.get("query")

        if not query:
            return templates.TemplateResponse(
                "mltimodal.html",
                {"request": request, "error": "Empty query"},
            )

        logger.info(f"Text query received: {query}")
        model = load_search_model()
        results = model.predict(query)

        return templates.TemplateResponse(
            "mltimodal.html",
            {"request": request, "results": results, "initial_query": query},
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return templates.TemplateResponse(
            "mltimodal.html",
            {"request": request, "error": str(e)},
        )


@app.post("/image", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Image received: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = load_search_model()
        results = model.predict(image)

        return JSONResponse({"results": results})
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        return JSONResponse({"error": str(e)})


@app.post("/api/search/text", tags=["API"])
async def api_text_search(query: str = Form(...)):
    try:
        model = load_search_model()
        results = model.predict(query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/search/image", tags=["API"])
async def api_image_search(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = load_search_model()
        results = model.predict(image)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
