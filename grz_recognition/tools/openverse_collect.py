"""Сбор кандидатов в реальную часть датасета через Openverse (api.openverse.org).

Openverse индексирует изображения под лицензиями Creative Commons (Flickr,
Wikimedia и др.). Запрашиваются только лицензии, совместимые с CC BY 4.0:
by, cc0, pdm. Результат дописывается в тот же JSONL, что и у
commons_collect.py (поля title, license, artist, page, url, ...).

python tools/openverse_collect.py --out work/commons/candidates.jsonl "taxi moscow" "такси" "маршрутка"
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

UA = "GRZ-dataset-collector/0.1 (olympiad research)"
API = "https://api.openverse.org/v1/images/"
LIC_NAMES = {"by": "CC BY", "cc0": "CC0", "pdm": "Public domain"}


def search(q, pages):
    for page in range(1, pages + 1):
        params = {"q": q, "license": "by,cc0,pdm", "page_size": 20, "page": page}
        url = API + "?" + urllib.parse.urlencode(params)
        for k in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                d = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except Exception:
                time.sleep(5 * (k + 1))
        else:
            return
        yield from d.get("results", [])
        if page >= d.get("page_count", 0):
            return
        time.sleep(1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--out", default="work/commons/candidates.jsonl")
    ap.add_argument("--pages", type=int, default=10)
    args = ap.parse_args()
    seen = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            d = json.loads(line)
            seen.add(d.get("url"))
            seen.add(d.get("page"))
    n = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for q in args.queries:
            k = 0
            for r in search(q, args.pages):
                if r["url"] in seen or r.get("foreign_landing_url") in seen:
                    continue
                seen.add(r["url"])
                lic = LIC_NAMES.get(r["license"], r["license"]) + (
                    f" {r['license_version']}" if r.get("license_version") and r["license"] == "by" else "")
                f.write(json.dumps({
                    "title": "Openverse:" + (r.get("title") or r["id"])[:100], "license": lic,
                    "artist": r.get("creator") or "", "page": r.get("foreign_landing_url") or r["url"],
                    "url": r["url"], "width": r.get("width"), "height": r.get("height"),
                    "mime": "image/jpeg", "category": "openverse:" + q, "provider": r.get("provider"),
                    "id": r["id"]}, ensure_ascii=False) + "\n")
                k += 1
            n += k
            print(f"{q}: +{k}", flush=True)
    print("всего добавлено:", n)


if __name__ == "__main__":
    main()
