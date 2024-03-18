"""custom HF data loader to load a large audio dataset from local
- run `reazon_downloader.py` first to download the desired data type (["tiny", "small", "medium", "large", "all"]) locally.
- credit: https://huggingface.co/datasets/reazon-research/reazonspeech/blob/main/reazonspeech.py

Example:
```
import os
from datasets import load_dataset

dataset = load_dataset(
    f"{os.getcwd()}/reazon_custom_loader.py",
    "tiny",
    split="train",
    trust_remote_code=True
)
```
"""
import os
from glob import glob

import datasets
from datasets.tasks import AutomaticSpeechRecognition

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

DATA_SIZE = ["tiny", "small", "medium", "large", "all"]


class ReazonSpeechConfig(datasets.BuilderConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ReazonSpeech(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ReazonSpeechConfig(name=name) for name in DATA_SIZE]
    DEFAULT_CONFIG_NAME = "tiny"
    DEFAULT_WRITER_BATCHDATA_SIZE = 256

    def _info(self):
        return datasets.DatasetInfo(
            task_templates=[AutomaticSpeechRecognition()],
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16000),
                    "transcription": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = f"{os.path.expanduser('~')}/.cache/reazon_manual_download/{self.config.name}"
        audio_files = glob(f"{data_dir}/*.tar")
        audio = [dl_manager.iter_archive(path) for path in audio_files]
        transcript_file = f"{data_dir}/{self.config.name}.{self.config.name}.tsv"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"audio_files": audio_files, "transcript_file": transcript_file, "audio": audio},
            ),
        ]

    def _generate_examples(self, audio_files, transcript_file, audio):

        # hashTable of a file and the associated transcript
        meta = {}
        with open(transcript_file, "r", encoding="utf-8") as fp:
            for line in fp:
                filename, transcription = line.rstrip("\n").split("\t")
                meta[filename] = transcription

        # iterator over audio
        for i, audio_single_dump in enumerate(audio):
            for filename, file in audio_single_dump:
                filename = filename.lstrip("./")
                if filename not in meta:  # skip audio without transcription
                    continue
                yield filename, {
                    "name": filename,
                    "audio": {"path": os.path.join(audio_files[i], filename), "bytes": file.read()},
                    "transcription": meta[filename],
                }
