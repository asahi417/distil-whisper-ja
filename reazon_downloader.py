"""Manual script to download reazonspeech dataset locally
- files are saved at your cache folder (`~/.cache/reazon_manual_download/{target_size}`)
- dataset https://huggingface.co/datasets/reazon-research/reazonspeech
"""
import argparse
import os
import urllib.request
from multiprocessing import Pool
from tqdm import tqdm

BASE_URL = "https://reazonspeech.s3.abci.ai/"
DATASET = {
    "tiny": {"tsv": 'v2-tsv/tiny.tsv', "audio": "v2/{:03x}.tar", "nfiles": 1},
    "small": {"tsv": 'v2-tsv/small.tsv', "audio": "v2/{:03x}.tar", "nfiles": 12},
    "medium": {"tsv": 'v2-tsv/medium.tsv', "audio": "v2/{:03x}.tar", "nfiles": 116},
    "large": {"tsv": 'v2-tsv/large.tsv', "audio": "v2/{:03x}.tar", "nfiles": 579},
    "all": {"tsv": 'v2-tsv/all.tsv', "audio": "v2/{:03x}.tar", "nfiles": 4096}
}


def dl(url, target_file):
    try:
        urllib.request.urlretrieve(url, target_file)
        return True
    except Exception as e:
        print(f"network error during downloading {target_file}")
        if os.path.exists(target_file):
            os.remove(target_file)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ReazonSpeech locally.')
    parser.add_argument('-t', '--target', default="tiny", help="tiny/small/medium/large/all", type=str)
    parser.add_argument('-p', '--pool', default=10, help="thread pool", type=int)
    arg = parser.parse_args()
    target_dir = f"{os.path.expanduser('~')}/.cache/reazon_manual_download/{arg.target}"
    os.makedirs(target_dir, exist_ok=True)

    # all urls to download
    urls = [BASE_URL + DATASET[arg.target]["audio"].format(idx) for idx in range(DATASET[arg.target]["nfiles"])]
    urls.append(BASE_URL + DATASET[arg.target]["tsv"])
    urls = [i for i in urls if not os.path.exists(f"{target_dir}/{arg.target}.{os.path.basename(i)}")]
    filenames = [f"{target_dir}/{arg.target}.{os.path.basename(i)}" for i in urls]
    print(f"Total files to download: {len(filenames)}")

    # start downloader
    print(f"Worker: {arg.pool}")
    with Pool(arg.pool) as pool:
        pool.starmap(dl, tqdm(zip(urls, filenames), total=len(filenames)))
