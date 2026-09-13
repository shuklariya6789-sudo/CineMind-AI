"""
Fetches recent Hindi-language movies from TMDB and appends them to
data/movie_database_final.csv in the same column format the app already
uses - so no other code needs to change.

SETUP (one-time):
    1. Free TMDB API key: https://www.themoviedb.org/settings/api
    2. pip install requests
    3. Paste your key into API_KEY below.

CONFIGURE:
    YEARS_BACK   - how many years back to search (default 3)
    MAX_PAGES    - TMDB returns 20 movies per page (default 15 pages = ~300 movies)

RUN:
    python scripts/add_hindi_movies.py

It skips any movie whose TMDB id is already present in the CSV, so it's
safe to re-run later to pick up newer releases - and safe to re-run if
it gets interrupted (e.g. by a dropped internet connection), since it
saves progress to the CSV every 20 movies instead of only at the end.

If you're on an unstable connection (mobile hotspot etc.) and every
request fails, try raising REQUEST_DELAY below to 0.5 or 1.0 - a flaky
connection often does better with fewer, slower requests than a burst
of fast ones.
"""
import csv
import shutil
import time
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = "9db8b62d0a53f0fd6af3289fcca8397f"
YEARS_BACK = 1
MAX_PAGES = 40  # ~20 movies per page; the loop stops early once TMDB
                # runs out of pages for the date range, so this is just
                # a safety ceiling, not a hard target
SAVE_EVERY = 20  # write progress to the CSV every N new movies
REQUEST_DELAY = 0.3  # seconds between requests - raise this on flaky wifi

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "movie_database_final.csv"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
DETAILS_URL = "https://api.themoviedb.org/3/movie/{id}"
GENRE_URL = "https://api.themoviedb.org/3/genre/movie/list"

# A single session with keep-alive turned OFF and built-in retries.
# On a flaky mobile-hotspot connection, reused ("keep-alive") sockets
# are often the ones that get killed mid-request - forcing a fresh
# connection per request avoids that.
SESSION = requests.Session()
SESSION.headers.update({"Connection": "close"})
_retry = Retry(
    total=6,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
SESSION.mount("https://", HTTPAdapter(max_retries=_retry, pool_maxsize=1))


def get_json(url, params, retries=6):
    """GET with its own manual retry loop on top of the session's
    built-in retries, so even a fully dropped connection (not just a
    bad status code) gets a few extra chances before we give up."""
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = min(2 ** attempt, 30)
            print(f"  [network hiccup: {exc.__class__.__name__}] "
                  f"retrying in {wait}s ({attempt}/{retries})...")
            time.sleep(wait)
    print("  [skipping this request after repeated failures]")
    return None


def get_genre_map():
    data = get_json(GENRE_URL, {"api_key": API_KEY}) or {}
    return {g["id"]: g["name"] for g in data.get("genres", [])}


def discover_hindi_movies():
    today = date.today()
    start = today.replace(year=today.year - YEARS_BACK)
    movies = []
    for page in range(1, MAX_PAGES + 1):
        params = {
            "api_key": API_KEY,
            "with_original_language": "hi",
            "sort_by": "primary_release_date.desc",
            "primary_release_date.gte": start.isoformat(),
            "primary_release_date.lte": today.isoformat(),
            "page": page,
        }
        data = get_json(DISCOVER_URL, params)
        if not data:
            continue
        movies.extend(data.get("results", []))
        if page >= data.get("total_pages", 1):
            break
        time.sleep(REQUEST_DELAY)
    return movies


def get_details(movie_id):
    params = {"api_key": API_KEY, "append_to_response": "credits"}
    return get_json(DETAILS_URL.format(id=movie_id), params)


def save_csv(rows, fieldnames):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if API_KEY == "PASTE_YOUR_TMDB_API_KEY_HERE":
        print("Set your TMDB API_KEY at the top of this script first.")
        return

    backup_path = CSV_PATH.with_suffix(".csv.bak2")
    if not backup_path.exists():
        shutil.copy(CSV_PATH, backup_path)
        print(f"Backup saved to {backup_path}")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else [
            "id", "title", "overview", "genres_clean", "cast_clean",
            "director", "vote_average", "vote_count", "popularity",
            "year", "poster_url",
        ]

    existing_ids = {str(r.get("id", "")).strip() for r in rows}

    print("Fetching genre list...")
    genre_map = get_genre_map()

    print(f"Searching TMDB for Hindi movies from the last {YEARS_BACK} years...")
    candidates = discover_hindi_movies()
    print(f"Found {len(candidates)} candidates. Fetching details...")

    added = 0
    for i, movie in enumerate(candidates, 1):
        tmdb_id = str(movie.get("id", ""))
        if not tmdb_id or tmdb_id in existing_ids:
            continue

        details = get_details(tmdb_id)
        if not details:
            continue  # couldn't fetch after retries - skip, move on

        credits = details.get("credits", {})
        cast_names = [c["name"] for c in credits.get("cast", [])[:8]]
        director_names = [
            c["name"] for c in credits.get("crew", [])
            if c.get("job") == "Director"
        ]
        genre_names = [
            genre_map.get(g["id"], "") for g in details.get("genres", [])
        ]
        release_date = details.get("release_date", "") or ""
        year = release_date[:4] if release_date else ""
        poster_path = details.get("poster_path")

        row = {
            "id": tmdb_id,
            "title": details.get("title", ""),
            "overview": details.get("overview", ""),
            "genres_clean": ", ".join(g for g in genre_names if g),
            "cast_clean": ", ".join(cast_names),
            "director": ", ".join(director_names),
            "vote_average": details.get("vote_average", 0),
            "vote_count": details.get("vote_count", 0),
            "popularity": details.get("popularity", 0),
            "year": year,
            "poster_url": f"{IMAGE_BASE}{poster_path}" if poster_path else "",
        }
        rows.append({k: row.get(k, "") for k in fieldnames})
        existing_ids.add(tmdb_id)
        added += 1

        if added % SAVE_EVERY == 0:
            save_csv(rows, fieldnames)
            print(f"...{i}/{len(candidates)} candidates processed "
                  f"({added} new movies added so far, progress saved)")

        time.sleep(REQUEST_DELAY)

    save_csv(rows, fieldnames)
    print(f"\nDone. Added {added} new Hindi movies.")
    print(f"Saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
