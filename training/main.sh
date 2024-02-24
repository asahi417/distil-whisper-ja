# Generate labels
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
DATASET_TYPE="tiny"
PROJECT_NAME="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_ORG="asahi417"

accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "reazon-research/reazonspeech" \
  --dataset_config_name "${DATASET_TYPE}" \
  --dataset_split_name "train" \
  --text_column_name "transcription" \
  --id_column_name "name" \
  --output_dir "./${PROJECT_NAME}" \
  --hub_model_id "${HF_ORG}/${PROJECT_NAME}" \
  --wandb_project "${PROJECT_NAME}" \
  --per_device_eval_batch_size 64 \
  --dtype "no" \
  --dataloader_num_workers 11 \
  --preprocessing_num_workers 16 \
  --logging_steps 500 \
  --max_label_length 128 \
  --language "ja" \
  --task "transcribe" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --streaming True \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --push_to_hub