"""Build-time fetch contributors/google scholar showcases, called from ``conf.py``

Tries to refresh contributors and scholar jsons, consumed client-side by ``_static/community/community.js``.

The fetching is best-effort and time-bounded so never hangs or breaks build. Any failure falls back to ``*.fallback.json``.
Fetching is also skipped when a fresh live file already exists, so repeated local builds don't re-hit the network.
Set ``DEEPINV_COMMUNITY_FETCH=0`` to always use the fallbacks.

The module can also be run as a script to (re)generate the committed fallbacks::

    python community_data.py --update-fallback
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

try:
    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


REPO = "deepinv/deepinv"
SCHOLAR_CITES = "2339544645882267464,1679399233449144578"  # deepinv paper
SCHOLAR_URL = f"https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites={SCHOLAR_CITES}&scipsc=&q=&scisbd=1"
SCHOLAR_MAX_PAPERS = 30
SCHOLAR_PAGE_SIZE = 10

EXCLUDED_CONTRIBUTORS = {"copilot", "github-actions[bot]", "claude"}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Refresh the live files at most once every TTL seconds (default 12h) so that local builds stay fast
CACHE_TTL_SECONDS = int(os.environ.get("DEEPINV_COMMUNITY_TTL", 12 * 3600))

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "_static", "community", "data")
)

VENUE_RULES = [
    ("arxiv", "arXiv"),
    ("international conference on learning representations", "ICLR"),
    ("computer vision and pattern recognition", "CVPR"),
]


### Google Scholar


def normalize_venue(venue: str) -> str:
    venue = re.sub(r"\s*[…\.]{2,}\s*", " ", venue).strip(" ,-–…")
    low = venue.lower()
    for needle, repl in VENUE_RULES:
        if needle in low:
            return repl
    if "ieee/cvf" in low and "international" not in low:
        return "CVPR"
    return venue


def format_authors(authors: str) -> str:
    truncated = "…" in authors or "..." in authors
    authors = authors.replace("…", "").replace("\xa0", " ")
    parts = [p.strip() for p in authors.split(",") if p.strip()]
    label = ", ".join(parts[:3])
    if truncated or len(parts) > 3:
        label += " et al."
    return label


def _parse_gs_a(text: str):
    """Parse Scholar's byline "AUTHORS - VENUE, YEAR - SOURCE"."""
    text = text.replace("\xa0", " ")
    segs = [s.strip() for s in text.split(" - ")]
    authors = segs[0] if segs else ""
    if len(segs) >= 3:
        middle = " - ".join(segs[1:-1])
    elif len(segs) == 2:
        middle = segs[1]
    else:
        middle = ""
    year = ""
    m = re.search(r"\b(19|20|21)\d{2}\b", middle)
    if m:
        year = m.group(0)
        venue = middle[: m.start()].rstrip(" ,")
    else:
        venue = middle
    return format_authors(authors), normalize_venue(venue), year


def parse_scholar_html(html: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    papers = []
    for div in soup.select("div.gs_ri"):
        h3 = div.select_one("h3.gs_rt")
        if not h3:
            continue
        title = re.sub(r"^(?:\s*\[[^\]]*\]\s*)+", "", h3.get_text().strip()).strip()
        if not title:
            continue
        link = h3.select_one("a")
        url = link["href"] if link and link.has_attr("href") else ""
        gs_a = div.select_one("div.gs_a")
        authors, venue, year = _parse_gs_a(gs_a.get_text() if gs_a else "")
        papers.append(
            {
                "title": title,
                "url": url,
                "authors": authors,
                "venue": venue,
                "year": year,
            }
        )
    return papers


def fetch_scholar(max_papers: int = SCHOLAR_MAX_PAPERS, timeout=(4, 6)) -> list[dict]:
    import requests

    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    )
    papers: list[dict] = []
    seen = set()
    for start in range(0, max_papers, SCHOLAR_PAGE_SIZE):
        resp = session.get(f"{SCHOLAR_URL}&start={start}", timeout=timeout)
        resp.raise_for_status()
        if "gs_captcha" in resp.text or "unusual traffic" in resp.text:
            raise RuntimeError("Google Scholar returned a CAPTCHA page")
        page = parse_scholar_html(resp.text)
        if not page:
            break
        for p in page:
            key = p["url"] or p["title"]
            if key not in seen:
                seen.add(key)
                papers.append(p)
        if len(page) < SCHOLAR_PAGE_SIZE:
            break
    if not papers:
        raise RuntimeError("No papers parsed from Google Scholar")
    return papers[:max_papers]


### GitHub contributors


def _github_headers(token):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "deepinv-docs",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_contributors(timeout=(4, 6), token=None, deadline: float | None = None):
    import requests

    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = _github_headers(token)

    resp = requests.get(
        f"https://api.github.com/repos/{REPO}/contributors",
        params={"per_page": 100, "anon": 0},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()
    base = []
    for c in raw:
        login = c.get("login", "")
        if not login or login.lower() in EXCLUDED_CONTRIBUTORS:
            continue
        base.append(
            {
                "login": login,
                "name": login,
                "location": None,
                "avatar_url": c.get("avatar_url"),
                "html_url": c.get("html_url"),
                "contributions": c.get("contributions"),
            }
        )
    if not base:
        raise RuntimeError("No contributors returned by GitHub")

    def enrich(entry):
        if deadline is not None and time.monotonic() > deadline:
            return entry
        try:
            r = requests.get(
                f"https://api.github.com/users/{entry['login']}",
                headers=headers,
                timeout=timeout,
            )
            r.raise_for_status()
            prof = r.json()
            entry["name"] = prof.get("name") or entry["login"]
            entry["location"] = prof.get("location")
            entry["avatar_url"] = prof.get("avatar_url") or entry["avatar_url"]
            entry["html_url"] = prof.get("html_url") or entry["html_url"]
        except Exception as exc:  # keep login-only on failure
            logger.info(
                "[community] profile enrich failed for %s: %s", entry["login"], exc
            )
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        base = list(pool.map(enrich, base))
    return base


### Orchestration


def _payload(kind: str, items):
    return {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": (
            SCHOLAR_URL
            if kind == "scholar"
            else f"https://github.com/{REPO}/graphs/contributors"
        ),
        kind: items,
    }


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_datajs(path, kind, payload):
    """Write the data as a <script>-loadable JS global.

    The showcases read ``window.DI_COMMUNITY[kind]`` instead of fetching JSON at
    page load, so they also work when the built HTML is opened directly from
    disk (``file://``), where ``fetch()`` is blocked by the browser."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.DI_COMMUNITY = window.DI_COMMUNITY || {};\n")
        f.write(
            "window.DI_COMMUNITY[%s] = %s;\n"
            % (json.dumps(kind), json.dumps(payload, ensure_ascii=False))
        )


def _is_fresh(path):
    try:
        if os.path.getsize(path) <= 2:
            return False
        age = time.time() - os.path.getmtime(path)
        return age < CACHE_TTL_SECONDS
    except OSError:
        return False


def _refresh_one(kind, data_dir, fetch_fn):
    """Refresh a single dataset into ``<kind>.data.js``, falling back to the
    committed ``<kind>.fallback.json`` snapshot."""
    live = os.path.join(data_dir, f"{kind}.data.js")
    fallback = os.path.join(data_dir, f"{kind}.fallback.json")

    if os.environ.get("DEEPINV_COMMUNITY_FETCH", "1") != "0" and not _is_fresh(live):
        try:
            items = fetch_fn()
            _write_datajs(live, kind, _payload(kind, items))
            logger.info("[community] refreshed %s.data.js (%d items)", kind, len(items))
            return
        except Exception as exc:
            logger.info(
                "[community] live fetch for %s failed, using fallback: %s", kind, exc
            )

    if _is_fresh(live):
        return  # keep the recent live file
    if os.path.exists(fallback):
        with open(fallback, encoding="utf-8") as f:
            _write_datajs(live, kind, json.load(f))
        logger.info("[community] using committed fallback for %s.data.js", kind)
    elif not os.path.exists(live):
        _write_datajs(live, kind, _payload(kind, []))


def generate_community_data(app=None, data_dir=None):
    data_dir = data_dir or (
        os.path.join(app.srcdir, "_static", "community", "data")
        if app is not None
        else DEFAULT_DATA_DIR
    )
    _refresh_one("scholar", data_dir, lambda: fetch_scholar())
    _refresh_one(
        "contributors",
        data_dir,
        lambda: fetch_contributors(deadline=time.monotonic() + 25),
    )


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update-fallback", action="store_true")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--html", nargs="*", help="local Scholar HTML pages to parse offline"
    )
    ap.add_argument("--profiles", help="local GitHub profiles JSON to use offline")
    args = ap.parse_args()

    if args.html is not None:
        papers = []
        for fn in args.html:
            with open(fn, encoding="utf-8", errors="replace") as f:
                papers += parse_scholar_html(f.read())
    else:
        papers = fetch_scholar()

    if args.profiles:
        with open(args.profiles, encoding="utf-8") as f:
            contributors = [
                c
                for c in json.load(f)
                if c.get("login", "").lower() not in EXCLUDED_CONTRIBUTORS
            ]
    else:
        contributors = fetch_contributors()

    suffix = "fallback.json" if args.update_fallback else "json"
    _write_json(
        os.path.join(args.data_dir, f"scholar.{suffix}"), _payload("scholar", papers)
    )
    _write_json(
        os.path.join(args.data_dir, f"contributors.{suffix}"),
        _payload("contributors", contributors),
    )
    print(
        f"Wrote {len(papers)} papers and {len(contributors)} contributors to {args.data_dir}"
    )


if __name__ == "__main__":
    sys.exit(_cli())
