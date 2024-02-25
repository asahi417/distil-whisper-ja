# Generate labels
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"

#DATASET_TYPE="tiny"
DATASET_TYPE="small"
#DATASET_TYPE="medium"
#DATASET_TYPE="large"
#DATASET_TYPE="all"

HF_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_ORG="asahi417"

accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "reazon-research/reazonspeech" \
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
  --output_dir "${HF_ALIAS}" \
  --wandb_project "wandb.${HF_ALIAS}" \
  --hub_model_id "${HF_ORG}/${HF_ALIAS}" \
  --push_to_hub
