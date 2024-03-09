import json
import os.path

from datasets import load_dataset
from tqdm import tqdm


def get_cumulative_max_min(
        current_value: float,
        cumulative_value: float,
        current_max: float,
        current_min: float,
):
    if current_max is None or current_value > current_max:
        current_max = current_value
    if current_min is None or current_value < current_min:
        current_min = current_value
    return cumulative_value + current_value, current_max, current_min


def dataset_statistics(data: str = "reazonspeech", data_type: str = "tiny"):

    if data == "reazonspeech":
        dataset = load_dataset(
            f"{os.getcwd()}/reazon_custom_loader.py",
            data_type,
            split="train",
            trust_remote_code=True
        )
    elif data == "ja_asr.jsut-basic5000":
        dataset = load_dataset(
            "asahi417/ja_asr.jsut-basic5000",
            split="test",
            trust_remote_code=True
        )
    elif data == "common_voice_8_0":
        dataset = load_dataset(
            "mozilla-foundation/common_voice_8_0",
            "ja",
            split="test",
            trust_remote_code=True
        )
    else:
        raise ValueError(f"unknown dataset {data}")
    iterator = iter(dataset)
    duration, duration_max, duration_min = 0, None, None
    amp_max, amp_max_max, amp_max_min = 0, None, None
    amp_min, amp_min_max, amp_min_min = 0, None, None
    amp_mean, amp_mean_max, amp_mean_min = 0, None, None
    data_size = 0
    for value in tqdm(iterator):
        ar = value['audio']['array']
        duration, duration_max, duration_min = get_cumulative_max_min(duration, len(ar) / value['audio']['sampling_rate'], duration_max, duration_min)
        amp_max, amp_max_max, amp_max_min = get_cumulative_max_min(amp_max, ar.max(), amp_max_max, amp_max_min)
        amp_min, amp_min_max, amp_min_min = get_cumulative_max_min(amp_min, ar.min(), amp_min_max, amp_min_min)
        amp_mean, amp_mean_max, amp_mean_min = get_cumulative_max_min(amp_mean, ar.mean(), amp_mean_max, amp_mean_min)
        data_size += 1
    return {
        "duration": [duration/data_size, duration_max, duration_min],
        "amp_max": [amp_max / data_size, amp_max_max, amp_max_min],
        "amp_min": [amp_min / data_size, amp_min_max, amp_min_min],
        "amp_mean": [amp_mean / data_size, amp_mean_max, amp_mean_min],
        "data_size": data_size
    }

if os.path.exists("data_statistics.json"):
    with open("data_statistics.json") as f:
        stats = json.load(f)
else:
    stats = {}

if "ja_asr.jsut-basic5000" not in stats:
    stats["ja_asr.jsut-basic5000"] = dataset_statistics("ja_asr.jsut-basic5000")
if "common_voice_8_0" not in stats:
    stats["common_voice_8_0"] = dataset_statistics("common_voice_8_0")
if "reazonspeech.tiny" not in stats:
    stats["reazonspeech.tiny"] = dataset_statistics(data_type="tiny")
if "reazonspeech.small" not in stats:
    stats["reazonspeech.small"] = dataset_statistics(data_type="small")
if "reazonspeech.medium" not in stats:
    stats["reazonspeech.medium"] = dataset_statistics(data_type="medium")
# if "reazonspeech.large" in stats:
#     stats["reazonspeech.large"] = dataset_statistics(data_type="large")
# if "reazonspeech.all" in stats:
#     stats["reazonspeech.all"] = dataset_statistics(data_type="all")

with open("data_statistics.json", "w") as f:
    json.dump(stats, f)



