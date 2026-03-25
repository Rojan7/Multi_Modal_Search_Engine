#Data Extractor
USER_AGENT = (
    "MultimodalSearchBot/0.1 "
    "(academic research; contact: lalitramanmishra@gmail.com)"
)
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.daraz.com.np/"
}



TIMEOUT = 15
MAX_CONCURRENT = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
base_urls=["https://www.daraz.com.np/#?"]
BLOCK_KEYWORDS = [
    "icon",
    "logo",
    "sprite",
    "favicon",
    "thumb",
    "avatar",
    "badge",
    "ads",
    "banner"
]

max_depth=10

#Training Pipeline
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

#Data Extractor
EXTRACTED_DIR_NAME : str = "Extracted_data_from_crwaling"


#Load model
MODEL_URL :str= "openai/clip-vit-base-patch32"
LOAD_MODEL_DIR : str = "Loaded_Model"
USE_HALF :bool = False


#Model trainer
lr=5e-6
batch_size=16
epochs=3
train_model_dir="Fine_Tuned_Model"




