"""
Fills in the missing `poster_url` column in movie_database_final.csv by
looking each movie up on TMDB (The Movie Database) by title + year.

SETUP (one-time):
    1. Create a free account at https://www.themoviedb.org/signup
    2. Go to Settings -> API -> request an API key (choose "Developer",
       it's free and instant for personal projects).
    3. pip install requests
    4. Paste your key into API_KEY below.

RUN:
    python scripts/backfill_posters.py

It edits data/movie_database_final.csv in place (a .bak backup of the
original is created first). Safe to stop and re-run - it skips rows
that already have a poster_url.
"""
import csv
import shutil
import time
from pathlib import Path

import requests

API_KEY = "9db8b62d0a53f0fd6af3289fcca8397f"

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "movie_database_final.csv"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"


def find_poster(title, year):
    params = {"api_key": API_KEY, "query": title}
    if year:
        params["year"] = year
    for attempt in range(3):
        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            break
        except requests.RequestException:
            if attempt == 2:
                return ""
            time.sleep(2 * (attempt + 1))

    if not results and year:
        # retry without the year filter - release-year mismatches
        # (region cuts, re-releases) are common
        params.pop("year")
        for attempt in range(3):
            try:
                resp = requests.get(SEARCH_URL, params=params, timeout=10)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                break
            except requests.RequestException:
                if attempt == 2:
                    return ""
                time.sleep(2 * (attempt + 1))

    if not results:
        return ""

    poster_path = results[0].get("poster_path")
    return f"{IMAGE_BASE}{poster_path}" if poster_path else ""


def main():
    if API_KEY == "PASTE_YOUR_TMDB_API_KEY_HERE":
        print("Set your TMDB API_KEY at the top of this script first.")
        return

    backup_path = CSV_PATH.with_suffix(".csv.bak")
    if not backup_path.exists():
        shutil.copy(CSV_PATH, backup_path)
        print(f"Backup saved to {backup_path}")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    updated, skipped, failed = 0, 0, 0
    for i, row in enumerate(rows, 1):
        existing = (row.get("poster_url") or "").strip()
        if existing:
            skipped += 1
            continue

        title = row.get("title", "").strip()
        year = str(row.get("year", "")).split(".")[0] if row.get("year") else ""
        poster = find_poster(title, year)

        if poster:
            row["poster_url"] = poster
            updated += 1
        else:
            failed += 1

        if i % 25 == 0:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"...{i}/{len(rows)} processed "
                  f"(updated {updated}, no match {failed}, progress saved)")

        time.sleep(0.15)  # stay comfortably under TMDB's rate limit

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Updated: {updated}  Already had a poster: {skipped}  "
          f"No match found: {failed}")
    print(f"Saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
