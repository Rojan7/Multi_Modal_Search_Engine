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
base_urls=["https://en.wikipedia.org/wiki/Kallang_Field"]
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
