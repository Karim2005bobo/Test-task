"""Сбор кандидатов в реальную часть датасета с Wikimedia Commons.

Обходит категории (с подкатегориями до заданной глубины), берёт метаданные
файлов и оставляет только лицензии, совместимые с публикацией датасета под
CC BY 4.0: CC0, Public Domain, CC BY (любой версии). SA/NC/ND отбрасываются.
Результат – JSONL с URL, автором, лицензией и ссылкой на страницу файла
(для полей source/license в meta.csv). Запросы идут с паузами, по правилам API.

python tools/commons_collect.py --out work/commons/candidates.jsonl --depth 3 \
    "Taxis in Russia" "Marshrutka in Russia" "Automobiles in Vladivostok"
python tools/commons_collect.py --download work/commons/candidates.jsonl --img-dir work/commons/img
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

UA = "GRZ-dataset-collector/0.1 (olympiad research; contact via GitHub)"
API = "https://commons.wikimedia.org/w/api.php"
OK_LICENSE = re.compile(r"^(cc0|public domain|pd|cc[- ]by[- ]\d(\.\d)?|cc[- ]by \d(\.\d)?|cc by \d(\.\d)?)", re.I)
BAD_LICENSE = re.compile(r"(sa|nc|nd)\b|share", re.I)


def get(params, pause=1.0):
    params = dict(params, format="json", formatversion=2)
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            data = json.load(urllib.request.urlopen(req, timeout=60))
            time.sleep(pause)
            return data
        except Exception:
            time.sleep(5 * (k + 1))
    raise RuntimeError("API недоступен: " + url)


def members(cat, kind):
    out, cont = [], {}
    while True:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": "Category:" + cat,
                 "cmtype": kind, "cmlimit": 500, **cont})
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


def file_info(titles):
    d = get({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
             "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 2048})
    res = []
    for p in d["query"]["pages"]:
        if "imageinfo" not in p:
            continue
        ii = p["imageinfo"][0]
        md = ii.get("extmetadata", {})
        lic = md.get("LicenseShortName", {}).get("value", "")
        artist = re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "")).strip()
        res.append({"title": p["title"], "license": lic, "artist": artist,
                    "page": ii.get("descriptionurl"), "url": ii.get("thumburl") or ii.get("url"),
                    "width": ii.get("width"), "height": ii.get("height"), "mime": ii.get("mime"),
                    "date": re.sub("<[^>]+>", "", md.get("DateTimeOriginal", {}).get("value", ""))[:40]})
    return res


def license_ok(lic):
    return bool(OK_LICENSE.match(lic.strip())) and not BAD_LICENSE.search(lic.replace("CC BY", ""))


def crawl(roots, depth, out):
    seen_cat, seen_file = set(), set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            seen_file.add(json.loads(line)["title"])
    queue = [(r.replace(" ", "_"), 0) for r in roots]
    with open(out, "a", encoding="utf-8") as f:
        while queue:
            cat, d = queue.pop(0)
            if cat in seen_cat:
                continue
            seen_cat.add(cat)
            files = [t for t in members(cat, "file") if t not in seen_file]
            kept = 0
            for i in range(0, len(files), 40):
                for info in file_info(files[i:i + 40]):
                    seen_file.add(info["title"])
                    if info["mime"] not in ("image/jpeg", "image/png") or not license_ok(info["license"]):
                        continue
                    info["category"] = cat
                    f.write(json.dumps(info, ensure_ascii=False) + "\n")
                    kept += 1
            f.flush()
            print(f"{'  ' * d}{cat}: {len(files)} файлов, подходит по лицензии {kept}", flush=True)
            if d < depth:
                queue += [(c[len("Category:"):], d + 1) for c in members(cat, "subcat")]


def download(cands, img_dir, max_side=2048):
    os.makedirs(img_dir, exist_ok=True)
    for line in open(cands, encoding="utf-8"):
        c = json.loads(line)
        if c.get("id"):  # запись из openverse_collect.py
            name = "ov_" + c["id"] + ".jpg"
        else:
            name = "wc_" + re.sub(r"[^\w.-]+", "_", c["title"][5:])[:120]
        path = os.path.join(img_dir, name)
        if os.path.exists(path):
            continue
        for k in range(5):
            try:
                req = urllib.request.Request(c["url"], headers={"User-Agent": UA})
                with open(path, "wb") as f:
                    f.write(urllib.request.urlopen(req, timeout=120).read())
                break
            except Exception:
                time.sleep(5 * (k + 1))
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--out", default="work/commons/candidates.jsonl")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--download", default=None, help="JSONL кандидатов для загрузки")
    ap.add_argument("--img-dir", default="work/commons/img")
    args = ap.parse_args()
    if args.download:
        download(args.download, args.img_dir)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        crawl(args.roots, args.depth, args.out)


if __name__ == "__main__":
    main()
