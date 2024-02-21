import json
import os.path

import pandas as pd
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


def dataset_statistics(
        data_name: str = "reazon-research/reazonspeech",
        data_type: str = "tiny",
        streaming: bool = True):
    dataset = load_dataset(data_name, data_type, trust_remote_code=True, split='train', streaming=streaming)
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

if os.path.exists("stats.json"):
    with open("stats.json") as f:
        stats = json.load(f)
else:
    stats = {}

stat_tiny = dataset_statistics(data_type="tiny")
stat_small = dataset_statistics(data_type="small")
stat_medium = dataset_statistics(data_type="medium")
stat_large = dataset_statistics(data_type="large")
stat_all = dataset_statistics(data_type="all")
with open("stats.json", "w") as f:
    json.dump({
        "tiny": stat_tiny,
        "small": stat_small,
        "medium": stat_medium,
        "large": stat_large,
        "all": stat_all,
    })



