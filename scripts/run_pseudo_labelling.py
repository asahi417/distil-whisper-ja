#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pseudo-labelling audio data using the Whisper model in preparation for distillation.
"""
import csv

# You can also adapt this script for your own pseudo-labelling tasks. Pointers for this are left as comments.
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union
import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import (
    DatasetDict,
    load_dataset,
)
from huggingface_hub import HfFolder, DatasetFilter, HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
# from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PREPROCESSING_ONLY = bool(int(os.getenv("PREPROCESSING_ONLY", 0)))

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


def safe_push(dataset_to_push, repo_name, config_name):
    while True:
        try:
            dataset_to_push.push_to_hub(repo_name, config_name=config_name)
            break
        except Exception:
            logger.warning(f"FAILED: push_to_hub on {repo_name} failed. wait 60 sec and retry soon...")
            time.sleep(60)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_dir_suffix: Optional[str] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="transcription",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    id_column_name: str = field(
        default="name",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    return_timestamps: bool = field(
        default=True,
        metadata={"help": "Whether to return the timestamps with the text."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual distillation. This argument should be set for multilingual distillation "
                "only. For English speech recognition, it should be left as `None`."
            )
        },
    )

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can statistics a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}
        file_ids = {"input_ids": [feature["file_id"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )
        file_ids_batch = self.processor.tokenizer.pad(
            file_ids,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["file_ids"] = file_ids_batch["input_ids"]

        return batch


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=None,
        project_dir=training_args.output_dir,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers(project_name="wandb_dummy")
    token = HfFolder().get_token()

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info(f"Training/evaluation parameters {training_args}")

    # 4. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(model_args.model_name_or_path, token=token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.model_name_or_path, token=token)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer, token=token
    )
    processor = WhisperProcessor.from_pretrained(model_args.model_name_or_path, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
    )
    model.eval()

    assert model.config.decoder_start_token_id is not None, "`config.decoder_start_token_id` is not correctly defined"
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We need to set the language and task ids for multilingual checkpoints
        tokenizer.set_prefix_tokens(
            language=data_args.language, task="transcribe", predict_timestamps=data_args.return_timestamps
        )
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    model_input_name = feature_extractor.model_input_names[0]

    # 5. Load dataset
    dataset_name = data_args.dataset_name
    # dataset_name_vectorized = f"{dataset_name}.vectorized"
    # fetch available datasets
    # api = HfApi()
    # d_filter = DatasetFilter(author=os.path.dirname(data_args.dataset_name))
    # dataset_list = [d.id for d in api.list_datasets(filter=d_filter)]
    # if dataset_name_vectorized in dataset_list:
    #     logger.info(f"load vectorized dataset {dataset_name_vectorized}")
    #     vectorized_datasets = DatasetDict({
    #         "train": load_dataset(
    #             dataset_name_vectorized,
    #             data_args.dataset_config_name,
    #             split="train",
    #             trust_remote_code=True,
    #             token=token,
    #             num_proc=data_args.preprocessing_num_workers,
    #             dataset_dir_suffix=data_args.dataset_dir_suffix
    #         )
    #     })
    # else:
    logger.info(f"load dataset {dataset_name}")
    raw_datasets = DatasetDict({
        "train": load_dataset(
            dataset_name,
            data_args.dataset_config_name,
            split="train",
            trust_remote_code=True,
            token=token,
            num_proc=data_args.preprocessing_num_workers,
            dataset_dir_suffix=data_args.dataset_dir_suffix
        )
    })
    assert data_args.audio_column_name in next(iter(raw_datasets.values())).column_names
    assert data_args.text_column_name in next(iter(raw_datasets.values())).column_names
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    def prepare_dataset(batch):
        # process audio
        sample = batch[data_args.audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        # process targets
        input_str = batch[data_args.text_column_name]
        batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
        # record the id of the sample as token ids
        batch["file_id"] = tokenizer(batch[data_args.id_column_name], add_special_tokens=False).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=raw_datasets_features,
        num_proc=data_args.preprocessing_num_workers,
        desc="preprocess dataset"
    )
        # safe_push(vectorized_datasets, dataset_name_vectorized, data_args.dataset_config_name)

    if PREPROCESSING_ONLY:
        return

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # 8. Define Training Schedule
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 9. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": data_args.return_timestamps
    }
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": data_args.language,
                "task": "transcribe",
            }
        )

    # 10. Prepare everything with accelerate
    model = accelerator.prepare(model)
    logger.info("***** Running Labelling *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(
        f"  Total eval batch size (w. parallel & distributed) = {training_args.per_device_eval_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  Predict labels with timestamps = {data_args.return_timestamps}")
    eval_preds = []
    eval_ids = []
    eval_loader = DataLoader(
        vectorized_datasets["train"],
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
    )
    eval_loader = accelerator.prepare(eval_loader)
    batches = tqdm(eval_loader, desc=f"Evaluating...", disable=not accelerator.is_local_main_process)
    output_csv = os.path.join(training_args.output_dir, "train-transcription.csv")
    generated_csv_data = []
    generated_file_ids = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="UTF8", newline="") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                generated_csv_data.append([
                    row[0], [int(i) for i in re.split(r"[\s\\n]+", row[1].replace('[', '').replace(']', ''))]]
                )
                generated_file_ids.add(row[0])

    for step, batch in enumerate(batches):
        # Filter already generated transcription
        file_ids = batch.pop("file_ids")
        valid_index = [n for n, i in enumerate(file_ids) if i not in generated_file_ids]
        if len(valid_index) == 0:
            continue
        batch = {k: [v[i] for i in valid_index] for k, v in batch.items()}
        file_ids = [file_ids[i] for i in valid_index]
        # Generate predictions and pad to max generated length
        generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
        generated_ids = generate_fn(batch["input_features"].to(dtype=torch.bfloat16), **gen_kwargs)
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
        # Gather all predictions and targets
        file_ids, generated_ids, labels = accelerator.gather_for_metrics((file_ids, generated_ids, batch["labels"]))
        eval_preds.extend(generated_ids.cpu().numpy())
        file_ids = tokenizer.batch_decode(file_ids, skip_special_tokens=True)
        eval_ids.extend(file_ids)
        # Log
        if step % training_args.logging_steps == 0 and step > 0:
            batches.write(f"Saving transcriptions: step {step}")
            accelerator.wait_for_everyone()
            csv_data = generated_csv_data + [[eval_ids[i], eval_preds[i]] for i in range(len(eval_preds))]
            with open(output_csv, "w", encoding="UTF8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_id", "whisper_transcript"])
                writer.writerows(csv_data)

    accelerator.wait_for_everyone()
    batches.write("Saving final transcriptions for.")
    csv_data = generated_csv_data + [[eval_ids[i], eval_preds[i]] for i in range(len(eval_preds))]
    with open(output_csv, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "whisper_transcript"])
        writer.writerows(csv_data)
    raw_datasets["train"] = raw_datasets["train"].add_column("whisper_transcript", eval_preds)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw_datasets.save_to_disk(training_args.output_dir, num_proc=data_args.preprocessing_num_workers)
        safe_push(raw_datasets, training_args.hub_model_id, data_args.dataset_config_name)
    accelerator.end_training()


if __name__ == "__main__":
    main()
