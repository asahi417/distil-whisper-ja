DATASET_TYPE="tiny"
MAX_STEPS=166
WARMUP_STEPS=50
SAVE_STEPS=50

DATASET_TYPE="small"
MAX_STEPS=1938
WARMUP_STEPS=500
SAVE_STEPS=500

DATASET_TYPE="medium"
MAX_STEPS=19348
WARMUP_STEPS=500
SAVE_STEPS=5000

#DATASET_TYPE="large"
#MAX_STEPS=
#WARMUP_STEPS=
#SAVE_STEPS=

#DATASET_TYPE="all"
#MAX_STEPS=
#WARMUP_STEPS=
#SAVE_STEPS=

##########
# Config #
##########
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"

TEACHER_MODEL="openai/whisper-large-v3"
HF_ORG="asahi417"
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"
huggingface-cli login

####################
# Download Dataset #
####################
python reazon_downloader.py --target "${DATASET_TYPE}"

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
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 1 \
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
cp reazon_custom_loader.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}"
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"


##########################
# Training Student Model #
##########################
rm -rf run_distillation.py
cp ../run_distillation.py ./
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}" \
  --train_dataset_config_name "${DATASET_TYPE}" \
  --language "ja" \
  --task "transcribe" \
  --train_split_name "train" \
  --text_column_name "transcription" \
  --save_steps ${SAVE_STEPS} \
  --warmup_steps ${WARMUP_STEPS} \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps "${MAX_STEPS}" \
  --wer_threshold 10 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 1 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --freeze_encoder \
  --push_to_hub


##########################
# Evaluate Student Model #
##########################
EVAL_DATASET="asahi417/ja_asr.common_voice_8_0"
#EVAL_DATASET="asahi417/ja_asr.jsut-basic5000"
accelerate launch run_short_form_eval.py \
  --model_name_or_path "${HF_ORG}/${HF_MODEL_ALIAS}" \
  --dataset_name "${EVAL_DATASET}" \
  --dataset_split_name "test" \
  --text_column_name "transcription" \
  --output_dir "eval/${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
  --per_device_eval_batch_size 256 \
  --dtype "bfloat16" \
  --dataloader_num_workers 64 \
  --generation_max_length 256 \
  --language "ja" \
  --task "transcribe" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
  --attn_type "flash_attn"


#####################################
# (Optional) Evaluate Teacher Model #
#####################################
EVAL_DATASET="asahi417/ja_asr.common_voice_8_0"
#EVAL_DATASET="asahi417/ja_asr.jsut-basic5000"
accelerate launch run_short_form_eval.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${EVAL_DATASET}" \
  --dataset_split_name "test" \
  --text_column_name "transcription" \
  --output_dir "eval/${TEACHER_MODEL##*/}" \
  --per_device_eval_batch_size 32 \
  --dtype "bfloat16" \
  --dataloader_num_workers 64 \
  --generation_max_length 256 \
  --language "ja" \
  --task "transcribe" \
  --wandb_project "wandb.${TEACHER_MODEL##*/}" \
  --attn_type "flash_attn"

