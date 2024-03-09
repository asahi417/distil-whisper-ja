# Japanese Distil-Whisper
This is an adaptation of [official distil-whisper training script](https://github.com/huggingface/distil-whisper/tree/main/training) into Japanese.
Following descriptions are taken from the original repository.

## Training Distil-Whisper
Reproducing the Distil-Whisper project requires four stages to be completed in successive order:

1. [Pseudo-labelling](#1-pseudo-labelling)
2. [Initialisation](#2-initialisation)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)

This README is partitioned according to the four stages. Each section provides a minimal example for running the
scripts used in the project. We will use a running example of distilling the Whisper model for Hindi speech recognition
on the Common Voice dataset. Note that this dataset only contains ~20 hours of audio data. Thus, it can be run extremely
quickly, but does not provide sufficient data to achieve optimal performance. We recommend training on upwards of 1000 
hours of data should you want to match the performance of Whisper on high-resource languages.

### Get Started  

- huggingface configuration
```bash
accelerate config
huggingface-cli login
```

- experiment configuration
```shell
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"

#DATASET_TYPE="tiny"
#DATASET_TYPE="small"
DATASET_TYPE="medium"
#DATASET_TYPE="large"
#DATASET_TYPE="all"

TEACHER_MODEL="openai/whisper-large-v3"
HF_ORG="asahi417"
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"
```

## 1. Pseudo-Labelling

The python script [`run_pseudo_labelling.py`](run_pseudo_labelling.py) is a flexible inference script that can be used
to generate pseudo-labels under a range of settings, including using both greedy and beam-search.

```bash
accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_config_name "${DATASET_TYPE}" \
  --dataset_split_name "train" \
  --text_column_name "transcription" \
  --id_column_name "name" \
  --per_device_eval_batch_size 50 \
  --dtype "bfloat16" \
  --dataloader_num_workers 8 \
  --preprocessing_num_workers 8 \
  --logging_steps 100 \
  --max_label_length 128 \
  --language "ja" \
  --task "transcribe" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --output_dir "${HF_DATASET_ALIAS}" \
  --wandb_project "wandb.${HF_DATASET_ALIAS}" \
  --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}" \
  --push_to_hub
rm -rf "${HF_DATASET_ALIAS}"
```


## 2. Initialisation

The script [`create_student_model.py`](create_student_model.py) can be used to initialise a small student model
from a large teacher model. When initialising a student model with fewer layers than the teacher model, the student is 
initialised by copying maximally spaced layers from the teacher, as per the [DistilBart](https://arxiv.org/abs/2010.13002)
recommendations.

First, we need to create a model repository on the Hugging Face Hub. This repository will contain all the required files 
to reproduce the training run, alongside model weights, training logs and a README.md card. You can either create a model 
repository directly on the Hugging Face Hub using the link: https://huggingface.co/new. Or, via the CLI, as we'll show here.
```bash
huggingface-cli repo create "${HF_MODEL_ALIAS}"
```

Let's clone the repository so that we can place our training script and model weights inside:

```bash
git lfs install
git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
```

We can now copy the relevant training scrips to the repository:
```bash
cp create_student_model.py "${HF_MODEL_ALIAS}"
cp run_distillation.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}" || exit
```

The following command demonstrates how to initialise a student model from the Whisper checkpoint, with all 32 encoder layer and 2 decoder layers. The 2 student decoder layers are copied from teacher layers 
1 and 32 respectively, as the maximally spaced layers:

```bash
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"
```

The initialised model will be saved to the sub-directory in our model repository. 

## 3. Training

The script [`run_distillation.py`](run_distillation.py) is an end-to-end script for loading multiple
datasets, a student model, a teacher model, and performing teacher-student distillation. It uses the loss formulation
from the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430), which is a weighted sum of the cross-entropy and 
KL-divergence loss terms.

The following command takes the Common Voice dataset that was pseudo-labelled in the first stage and trains the 
2-layer decoder model intialised in the previous step. We pass the local path to the pseudo-labelled Common Voice dataset
(`../common_voice_13_0_hi_pseudo_labelled`), which you can change to the path where your local pseudo-labelled dataset is 
saved.

In this example, we will combine the train and validation splits to give our training set, and evaluate on the test split 
only. This is purely to demonstrate how to combine multiple pseudo-labelled datasets for training, rather than recommended 
advice for defining train/validation splits. We advise that you train on the train splits of your dataset, evaluate and 
tune hyper-parameters on the validation split, and only test the final checkpoint on the test split. Note how multiple 
training datasets and splits can be loaded by separating the dataset arguments by `+` symbols. Thus, the script generalises 
to any number of training datasets.

```bash
#!/usr/bin/env bash

accelerate launch run_distillation.py \
  --model_name_or_path "./distil-large-v2-init" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_name "../common_voice_13_0_hi_pseudo_labelled+../common_voice_13_0_hi_pseudo_labelled" \
  --train_dataset_config_name "hi+hi" \
  --train_split_name "train+validation" \
  --text_column_name "sentence+sentence" \
  --train_dataset_samples "10+5" \
  --eval_dataset_name "../common_voice_13_0_hi_pseudo_labelled" \
  --eval_dataset_config_name "hi" \
  --eval_split_name "test" \
  --eval_text_column_name "sentence" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 5000 \
  --wer_threshold 10 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --streaming False \
  --push_to_hub

```

The above training script will take approximately 1 hour to complete on an 80 GB A100 GPU and yield a final WER of 31%.
This is reasonable for 1000 training steps and just 15 hours of un-filtered training data, but 12% higher than the error rate of the 
pre-trained model. As mentioned above, using upwards of 1000 hours of data and training for 10k steps will likely yield
more competitive performance. For the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430), we trained on 21k hours
of audio data for 80k steps. We found that upwards of 13k hours of audio data was required to reach convergence on English 
ASR (see Section 9.2 of the [paper](https://arxiv.org/abs/2311.00430)), so the more data you have, the better!

Scaling to multiple GPUs using [distributed data parallelism (DDP)](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)
is trivial: simply run `accelerate config` and select the multi-GPU option, specifying the IDs of the GPUs you wish to use. The 
above script can then be run using DDP with no code changes. 

Training logs will be reported to TensorBoard and WandB, provided the relevant packages are available. An example of a 
saved checkpoint pushed to the Hugging Face Hub can be found here: [sanchit-gandhi/distil-whisper-large-v2-hi](https://huggingface.co/sanchit-gandhi/distil-whisper-large-v2-hi).

There are a few noteworthy arguments that can be configured to give optimal training performance:
1. `train_dataset_samples`: defines the number of training samples in each dataset. Used to calculate the sampling probabilities in the dataloader. A good starting point is setting the samples to the number of hours of audio data in each split. A more refined strategy is setting it to the number of training samples in each split, however this might require downloading the dataset offline to compute these statistics.
2. `wer_threshold`: sets the WER threshold between the normalised pseudo-labels and normalised ground truth labels. Any samples with WER > `wer_threshold` are discarded from the training data. This is beneficial to avoid training the student model on pseudo-labels where Whisper hallucinated or got the predictions grossly wrong.
3. `freeze_encoder`: whether to freeze the entire encoder of the student model during training. Beneficial when the student encoder is copied exactly from the teacher encoder. In this case, the encoder hidden-states from the teacher model are re-used for the student model. Stopping the gradient computation through the encoder and sharing the encoder hidden-states provides a significant memory saving, and can enable up to 2x batch sizes. 
4. `dtype`: data type (dtype) in which the model computation should be performed. Note that this only controls the dtype of the computations (forward and backward pass), and not the dtype of the parameters or optimiser states.
5. `lr_scheduler_stype`: defines the learning rate schedule, one of `constant_with_warmup` or `linear`. When experimenting with a training set-up or training for very few steps (< 5k), using `constant_with_warmup` is typically beneficial, since the learning rate remains high over the short training run. When performing long training runs (> 5k), using a `linear` schedule generally results in superior downstream performance of the distilled model.
6. `streaming`: whether or not to use Datasets' streaming mode. Recommended for large datasets, where the audio data can be streamed from the Hugging Face Hub with no disk space requirements.
7. `timestamp_probability`: the per-sample probability for retaining timestamp tokens in the labels (should they contain them). Retaining some portion of timestamp tokens in the training data is required to ensure the distilled model can predict timestamps at inference time. In our experiments, we found that training on timestamps with high-probability hurts the distilled model's transcription performance. Thus, we recommend setting this to a value below 0.5. Typically, a value of 0.2 works well, giving good transcription and timestamp performance.
8. `condition_on_prev_probability`: the per-sample probability for conditioning on previous labels. Conditioning on previous tokens is required to ensure the distilled model can be used with the "sequential" long-form transcription algorithm at inference time. We did not experiment with this parameter, but found a value of 0.1 to provide adequate performance. OpenAI pre-trained Whisper on with a 50% probability for conditioning on previous tokens. Thus, you might wish to try higher values.

## 4. Evaluation

There are two types of evaluation performed in Distil-Whisper:
1. Short form: evaluation on audio samples less than 30s in duration. Examples include typical ASR test sets, such as the LibriSpeech validation set.
2. Long form: evaluation on audio samples longer than 30s in duration. Examples include entire TED talks or earnings calls.

Both forms of evaluation are performed using the *word-error rate (WER)* metric.

### Short Form

The script [`run_short_form_eval.py`](run_short_form_eval.py) can be used to evaluate a trained student model over 
multiple validation sets. The following example demonstrates how to evaluate the student model trained in the previous 
step on the Common Voice `test` set and also the FLEURS `test` set. Again, it leverages streaming mode to 
bypass the need to download the data offline:

```bash
#!/usr/bin/env bash

accelerate launch run_short_form_eval.py \
  --model_name_or_path "./" \
  --dataset_name "../common_voice_13_0_hi_pseudo_labelled+google/fleurs" \
  --dataset_config_name "hi+hi_in" \
  --dataset_split_name "test+test" \
  --text_column_name "sentence+transcription" \
  --output_dir "./" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --report_to "wandb" \
  --generation_max_length 128 \
  --language "hi" \
  --attn_type "flash_attn" \
  --streaming
```

It is particularly important to evaluate the final model on data that is *out-of-distribution (OOD)* with the training data. 
Evaluating on OOD data provides insight as to how well the distilled model is likely to generalise to different audio 
distributions at inference time. In this example, Common Voice is *in-distribution (ID)*, since it is taken from the same 
distribution as the Common Voice training set, whereas FLEURS is OOD, since it is not used as part of the training set.

### Long Form

Long form evaluation runs on the premise that a single long audio file can be *chunked* into smaller segments and 
inferred in parallel. The resulting transcriptions are then joined at the boundaries to give the final text prediction. 
A small overlap (or *stride*) is used between adjacent segments to ensure a continuous transcription across chunks.

This style of chunked inference is performed using the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines)
class, which provides a wrapper around the [`.generate`](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate) 
function for long-form inference.

The script [`run_long_form_eval.py`](run_long_form_eval.py) can be used to evaluate the trained student model on an 
arbitrary number of long-form evaluation sets. Since we don't have a long-form validation set for Hindi to hand, we'll
evaluate the teacher model on the TED-LIUM validation set in this example:

```bash
#!/usr/bin/env bash

python run_long_form_eval.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "distil-whisper/tedlium-long-form" \
  --dataset_config_name "all" \
  --dataset_split_name "validation" \
  --text_column_name "text" \
  --output_dir "./" \
  --per_device_eval_batch_size 64 \
  --chunk_length_s 30 \
  --language "en" \
  --return_timestamps \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming
```

The argument `chunk_length_s` controls the length of the chunked audio samples. It should be set to match the typical
length of audio the student model was trained on. If unsure about what value of `chunk_length_s` is optimal for your case,
it is recommended to run a *sweep* over all possible values. A template script for running a [WandB sweep](https://docs.wandb.ai/guides/sweeps) 
can be found under [`run_chunk_length_s_sweep.yaml`](flax/long_form_transcription_scripts/run_chunk_length_s_sweep.yaml).
