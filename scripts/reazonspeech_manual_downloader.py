"""Manual script to download reazonspeech dataset locally
- files are saved at your cache folder (`~/.cache/reazon_manual_download/{target_size}`)
- dataset https://huggingface.co/datasets/reazon-research/reazonspeech
"""
import argparse
import os
import urllib.request
from multiprocessing import Pool
from tqdm import tqdm
import tarfile

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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


def get_broken_files(target_files):
    broken_files = []
    for i in tqdm(target_files):
        try:
            with tarfile.open(i) as t:
                t.extractall()
        except tarfile.ReadError:
            print(f"broken file found: {i}")
            broken_files.append(i)
    print(f"{len(broken_files)} broken files found.")
    return broken_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ReazonSpeech locally.')
    parser.add_argument('-t', '--target', default="tiny", help="tiny/small/medium/large/all", type=str)
    parser.add_argument('-p', '--pool', default=10, help="thread pool", type=int)
    parser.add_argument('-s', '--start-que', default=None, help="thread pool", type=int)
    parser.add_argument('-e', '--end-que', default=None, help="thread pool", type=int)
    arg = parser.parse_args()
    target_dir = f"{os.path.expanduser('~')}/.cache/reazon_manual_download/{arg.target}"
    os.makedirs(target_dir, exist_ok=True)

    # all urls to download
    files = list(range(DATASET[arg.target]["nfiles"]))
    if arg.start_que is not None:
        assert arg.end_que is not None
        files = files[arg.start_que:arg.end_que]
    urls = [BASE_URL + DATASET[arg.target]["audio"].format(idx) for idx in files]
    urls.append(BASE_URL + DATASET[arg.target]["tsv"])
    urls = [i for i in urls if not os.path.exists(f"{target_dir}/{arg.target}.{os.path.basename(i)}")]
    filenames = [f"{target_dir}/{arg.target}.{os.path.basename(i)}" for i in urls]
    print(f"Total files to download: {len(filenames)}")

    while True:
        # start downloader
        print(f"Worker: {arg.pool}")
        if arg.pool == 1:
            for _url, _file in tqdm(zip(urls, filenames), total=len(filenames)):
                dl(_url, _file)
        else:
            with Pool(arg.pool) as pool:
                pool.starmap(dl, tqdm(zip(urls, filenames), total=len(filenames)))
        print("download complete")

        # check the tar files
        filenames = get_broken_files(filenames)
        if len(filenames) == 0:
            break
        for i in filenames:
            os.remove(i)
        print(f"retry downloading {len(filenames)} files")
