"""Download Bank of Jamaica monetary policy press releases from the BOJ website.

The BOJ site runs WordPress, so the public REST API lists every post in the
"Monetary Policy Decisions" category. We keep the policy decision press
releases (not schedules, summaries or awards) and save their plain text.
"""
import html
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://boj.org.jm/wp-json/wp/v2"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "statements.jsonl"
KEEP = re.compile(r"^(monetary-policy-press-release|monetary-policy-announcement|bank-of-jamaica-holds-policy-rate)")
HEADERS = {"User-Agent": "boj-policy-tone research project (github.com/Soulkinggja)"}


def get(url, **params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def list_posts():
    posts = {}
    cats = get(f"{API}/categories", search="monetary", _fields="id,slug").json()
    cat_ids = [c["id"] for c in cats if c["slug"] == "monetary-policy"]
    queries = [dict(categories=cid) for cid in cat_ids]
    queries += [dict(search=q) for q in ("monetary policy press release", "monetary policy announcement", "policy rate")]
    for q in queries:
        page = 1
        while True:
            r = requests.get(f"{API}/posts", params={**q, "per_page": 100, "page": page,
                                                     "_fields": "id,date,slug,link"},
                             headers=HEADERS, timeout=60)
            if r.status_code == 400:  # past the last page
                break
            r.raise_for_status()
            batch = r.json()
            for p in batch:
                posts[p["id"]] = p
            if len(batch) < 100:
                break
            page += 1
    keep = [p for p in posts.values() if KEEP.match(p["slug"]) and "schedule" not in p["slug"]]
    return sorted(keep, key=lambda p: p["date"])


def text_of(post_id):
    content = get(f"{API}/posts/{post_id}", _fields="content").json()["content"]["rendered"]
    soup = BeautifulSoup(content, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    text = html.unescape(text).replace(" ", " ")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def main():
    try:
        posts = list_posts()
    except Exception as e:  # the BOJ site sometimes blocks cloud servers
        print(f"BOJ site not reachable from this machine ({e}); keeping the saved statements")
        return
    if OUT.exists():
        saved = {json.loads(l)["id"] for l in OUT.open()}
        if {p["id"] for p in posts} <= saved:
            print("no new statements")
            return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for p in posts:
            rec = {"id": p["id"], "date": p["date"][:10], "slug": p["slug"], "url": p["link"],
                   "text": text_of(p["id"])}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved {len(posts)} statements, {posts[0]['date'][:10]} to {posts[-1]['date'][:10]}")


if __name__ == "__main__":
    main()
