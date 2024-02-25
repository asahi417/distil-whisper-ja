"""Manual script to download reazonspeech dataset locally
- files are saved at your cache folder (`~/.cache/reazon_manual_download/{target_size}`)
- dataset https://huggingface.co/datasets/reazon-research/reazonspeech
"""
import os
import urllib.request
from time import time
from os.path import expanduser


# target = "tiny"
# target = "small"
# target = "medium"
target = "large"
# target = "all"

base_url = "https://reazonspeech.s3.abci.ai/"
dataset = {
    "tiny": {"tsv": 'v2-tsv/tiny.tsv',   "audio": "v2/{:03x}.tar", "nfiles": 1},
    "small": {"tsv": 'v2-tsv/small.tsv',  "audio": "v2/{:03x}.tar", "nfiles": 12},
    "medium": {"tsv": 'v2-tsv/medium.tsv', "audio": "v2/{:03x}.tar", "nfiles": 116},
    "large": {"tsv": 'v2-tsv/large.tsv',  "audio": "v2/{:03x}.tar", "nfiles": 579},
    "all": {"tsv": 'v2-tsv/all.tsv',    "audio": "v2/{:03x}.tar", "nfiles": 4096}
}

urls = [base_url + dataset[target]["audio"].format(idx) for idx in range(dataset[target]["nfiles"])]
urls.append(base_url + dataset[target]["tsv"])
target_dir = f"{expanduser('~')}/.cache/reazon_manual_download/{target}"
os.makedirs(target_dir, exist_ok=True)
start_total = time()
for n, url in enumerate(urls):
    start = time()
    print(f"{n + 1}/{len(urls)}: {url}")
    target_file = f"{target_dir}/{target}.{os.path.basename(url)}"
    if os.path.exists(target_file):
        continue
    try:
        urllib.request.urlretrieve(url, target_file)
        print(f"- tmp: {time() - start} sec")
        print(f"- total: {time() - start_total} sec")
    except Exception as e:
        print(f"network error during downloading {target_file}")
        if os.path.exists(target_file):
            os.remove(target_file)
        exit()
print(f"total: {time()-start_total} sec")
