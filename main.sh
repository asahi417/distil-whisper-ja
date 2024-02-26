export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"

DATASET_TYPE="tiny"
#DATASET_TYPE="small"
#DATASET_TYPE="medium"
#DATASET_TYPE="large"
#DATASET_TYPE="all"
TEACHER_MODEL="openai/whisper-large-v3"
HF_ORG="asahi417"
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"

###################
# Generate Labels #
###################

accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${PWD}/reazon_custom_loader.py" \
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


############################
# Initialize Student Model #
############################
huggingface-cli repo create "${HF_MODEL_ALIAS}"
git lfs install
git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
cp create_student_model.py "${HF_MODEL_ALIAS}"
cp run_distillation.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}" || exit
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"


##########################
# Training Student Model #
##########################
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${PWD}/reazon_custom_loader.py" \
  --train_dataset_config_name "${DATASET_TYPE}" \
  --language "ja" \
  --task "transcribe" \
  --train_split_name "train" \
  --text_column_name "transcription" \
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
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --push_to_hub