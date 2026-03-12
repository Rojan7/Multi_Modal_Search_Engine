"""
Crawler module for Multimodal Search Engine
- Text + Image crawling
- Pluggable sources
- Async, rate-limited
- Metadata-first design
"""

import asyncio
import aiohttp
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from src.constants import USER_AGENT, TIMEOUT, IMAGE_EXTENSIONS,MAX_CONCURRENT,base_urls,BLOCK_KEYWORDS, max_depth
from src.exception import MyException
from src.logger import logger
import sys




def url_hash(url: str) -> str:
    logger.debug("Entered the url_has method")
    try:
        return hashlib.sha256(url.encode()).hexdigest() #converts url into 64 hexa chars fix length so it is fast to compare
    except Exception as e:
        raise MyException(e,sys)


def is_image(url: str) -> bool:
    logger.debug("Entered the method is image")
    try:
        return Path(urlparse(url).path).suffix.lower() in IMAGE_EXTENSIONS
    except Exception as e:
        raise MyException(e,sys)


class WebCrawler:
    def __init__(self, base_urls: list[str], output_dir="data/raw"):
        self.base_urls = base_urls
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visited = set()
        self.sem = asyncio.Semaphore(MAX_CONCURRENT)
        self.max_depth=max_depth
        

    async def fetch(self, session: aiohttp.ClientSession, url: str):
        logger.debug("Entered teh method fetch of class WebCrawler")
        async with self.sem: #concept of semaphore to liimit the maximum requests in event loop

            try:
                logger.debug(f"Fetching URL: {url}")
                async with session.get(url, timeout=TIMEOUT) as resp:
                    if resp.status == 200:
                        return await resp.text(), resp.headers.get("Content-Type", "")
                    else:
                        logger.debug(f"Non-200 status {resp.status} for {url}")
            except asyncio.TimeoutError as e:
                raise MyException(e,sys)
            except aiohttp.ClientError as e:
                raise MyException(e,sys)
            except Exception as e:
                raise MyException(e,sys)
        return None, None

    async def download_image(self, session, url):
        logger.debug("Entered download_image method of class WebCrawller")
        async with self.sem:
            try:
                logger.debug(f"Downloading image: {url}")
                async with session.get(url, timeout=TIMEOUT) as resp:
                    if resp.status == 200:
                        img_bytes = await resp.read()
                        fname = url_hash(url) + Path(url).suffix
                        path = self.output_dir / "images" / fname
                        path.parent.mkdir(parents=True,exist_ok=True)
                        path.write_bytes(img_bytes)
                        return {
                            "url": url,
                            "path": str(path),
                        }
                    else:
                        logger.debug(f"Image download failed ({resp.status}) for {url}")
            except asyncio.TimeoutError as e:
                raise MyException(e,sys)
                
            except aiohttp.ClientError as e:
                raise MyException(e,sys)
            except Exception as e:
                raise MyException(e,sys)
        return None

    async def crawl_page(self, session, url,depth):

        logger.debug("Entered crawl_pages method of class WebCrawller")
        if url in self.visited or depth>max_depth:
            logger.debug(f"Already visited: {url}")
            return None, [], []

        self.visited.add(url)
        logger.debug(f"Crawling page: {url}")

        try:
            html, content_type = await self.fetch(session, url)
            if not html or "text/html" not in content_type:
                logger.debug(f"Skipping non-HTML or empty content: {url}")
                return None, [], []

            soup = BeautifulSoup(html, "html.parser")

            # Extract page links
            page_urls = [a.get('href') for a in soup.find_all('a') if a.get('href')]
            page_urls = [urljoin(url, u) for u in page_urls]

            # Extract full page text
            text = soup.get_text(" ", strip=True)
            text_meta = {
                "url": url,
                "text": text,
                "type": "text",
            }

      
            images = []

            for img in soup.find_all("img"):
                src = img.get("src")
                if not src:
                    continue

                img_url = urljoin(url, src)

                if not is_image(img_url):
                    continue

                caption = None


                parent_figure = img.find_parent("figure")
                if parent_figure:
                    figcaption = parent_figure.find("figcaption")
                    if figcaption:
                        caption = figcaption.get_text(" ", strip=True)


                if not caption and img.get("alt"):
                    caption = img.get("alt").strip()


                if not caption and img.get("title"):
                    caption = img.get("title").strip()

                if not caption:
                    next_tag = img.find_next_sibling()
                    if next_tag and next_tag.name == "p":
                        caption = next_tag.get_text(" ", strip=True)

                # Filter junk captions
                if caption and len(caption) > 5:
                    images.append({
                        "url": img_url,
                        "caption": caption,
                        "page_url": url,
                        "type": "image"
                    })

            return text_meta, images, page_urls

        except Exception as e:
            raise MyException(e, sys)
        
    async def run(self):
        logger.debug("Entered run method of class webcrawler")

        try:
            headers = {"User-Agent": USER_AGENT}

            async with aiohttp.ClientSession(headers=headers) as session:

                queue = [(url, 0) for url in self.base_urls]

                all_text = []
                all_images = []

                while queue:

                    url, depth = queue.pop(0)

                    if url in self.visited:
                        continue

                    text_meta, images, page_urls = await self.crawl_page(session, url, depth)

                    if text_meta:
                        all_text.append(text_meta)

                    all_images.extend(images)

                    # Add new links to queue
                    for link in page_urls:
                        parsed_seed = urlparse(self.base_urls[0]).netloc
                        parsed_link = urlparse(link).netloc

                        # Domain restriction
                        if parsed_seed in parsed_link:
                            queue.append((link, depth + 1))

                # Download images
                img_tasks = [self.download_image(session, img["url"]) for img in all_images]
                img_meta = await asyncio.gather(*img_tasks)

                final_images = []

                for meta, downloaded in zip(all_images, img_meta):
                    if downloaded:
                        final_images.append({
                            "url": meta["url"],
                            "caption": meta["caption"],
                            "page_url": meta["page_url"],
                            "local_path": downloaded["path"],
                            "type": "image"
                        })

            return all_text, final_images

        except Exception as e:
            raise MyException(e, sys)
        
        
if __name__ == "__main__":
    seed_urls = ["https://en.wikipedia.org/wiki/Kallang_Field"]
    crawler = WebCrawler(seed_urls)
    image_data = asyncio.run(crawler.run())
    print(f" Images: {len(image_data)}")
