import argparse
import re
from functools import partial
import datasets
import evaluate
from datasets import DatasetDict, load_dataset
from transformers import WhisperTokenizerFast
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer


def main():

    parser = argparse.ArgumentParser(description='Download ReazonSpeech locally.')
    parser.add_argument('-p', '--push-to', default=None, type=str)
    parser.add_argument('-d', '--dataset', default="asahi417/whisper_transcriptions.reazonspeech.large", type=str)
    parser.add_argument('--dataset-config', default="large", type=str)
    parser.add_argument('--wer-threshold', default=10, type=float)
    parser.add_argument('-c', '--text-column', default="transcription", type=str)
    parser.add_argument('-a', '--audio-column', default="audio", type=str)
    parser.add_argument('-s', '--sampling-rate', default=16000, help="", type=int)
    parser.add_argument('--split', default="train", help="", type=str)
    parser.add_argument('--language', default="la", help="", type=str)
    parser.add_argument('--model', default="openai/whisper-large-v3", help="", type=str)
    parser.add_argument('-p', '--preprocessing-num-workers', default=64, help="thread pool", type=int)
    arg = parser.parse_args()

    dataset = load_dataset(arg.dataset_name, arg.dataset_config_name, split=arg.split, trust_remote_code=True)
    dataset = dataset.cast_column("audio", datasets.features.Audio(arg.sampling_rate))
    columns_to_keep = {"audio", "text", "whisper_transcript"}
    dataset = dataset.rename_column(arg.text_column, "text")
    dataset_features = dataset.features.keys()
    raw_datasets = DatasetDict()
    raw_datasets["train"] = dataset.remove_columns(set(dataset_features - columns_to_keep))
    raw_datasets = raw_datasets.cast_column(arg.audio_column, datasets.features.Audio(sampling_rate=arg.sampling_rate))

    metric = evaluate.load("wer")
    if arg.language != "en":
        normalizer = BasicTextNormalizer()
    else:
        tokenizer = WhisperTokenizerFast.from_pretrained(arg.model)
        normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    def is_wer_in_range(ground_truth, whisper_transcript):
        norm_ground_truth = normalizer(ground_truth)
        if (
            isinstance(whisper_transcript, str)
            and whisper_transcript.startswith("[")
            and whisper_transcript.endswith("]")
        ):
            whisper_transcript = re.findall(r"\d+", whisper_transcript)
            whisper_transcript = [int(token) for token in whisper_transcript]
        if isinstance(whisper_transcript, list):
            whisper_transcript = tokenizer.decode(whisper_transcript, skip_special_tokens=True)
        if len(norm_ground_truth) > 0 and whisper_transcript is not None:
            norm_whisper_transcript = normalizer(whisper_transcript)
            wer = 100 * metric.compute(predictions=[norm_whisper_transcript], references=[norm_ground_truth])
            return wer < arg.wer_threshold
        else:
            # filter automatically since we can't know the WER
            return False

    filter_by_wer_threshold = partial(
        raw_datasets["train"].filter, function=is_wer_in_range, input_columns=["text", "whisper_transcript"],
    )
    raw_datasets["train"] = filter_by_wer_threshold(
        num_proc=arg.preprocessing_num_workers, desc="filtering train dataset by wer"
    )
    repo_name = f"{arg.dataset}.wer_{arg.wer_threshold}" if arg.push_to is None else arg.push_to
    raw_datasets.push_to_hub(repo_name, config_name=arg.dataset_config_name)


if __name__ == "__main__":
    main()
