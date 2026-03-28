"""
Run the full CUNY scrape + ingestion pipeline.

Usage:
    python scripts/run_scrape.py                    # scrape all schools
    python scripts/run_scrape.py --school baruch    # scrape one school
    python scripts/run_scrape.py --max-pages 100    # limit pages per school
"""
import asyncio
import argparse
import logging
import sys

sys.path.insert(0, ".")

from config.settings import settings
from src.scraper.spider import crawl_school
from src.ingestion.pipeline import ingest_pages
from src.ingestion.embedder import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(school_filter: str | None, max_pages: int):
    embedding_model = get_embedding_model()
    schools = settings.cuny_senior_colleges

    if school_filter:
        if school_filter not in schools:
            logger.error(f"Unknown school: {school_filter}. Options: {list(schools.keys())}")
            return
        schools = {school_filter: schools[school_filter]}

    all_pages = []
    for school, url in schools.items():
        logger.info(f"Crawling {school}: {url}")
        pages = await crawl_school(school, url, max_pages=max_pages)
        all_pages.extend(pages)
        logger.info(f"Collected {len(pages)} pages from {school}")

    logger.info(f"Total pages: {len(all_pages)}. Starting ingestion...")
    await ingest_pages(all_pages, embedding_model)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(main(args.school, args.max_pages))
