DATASET_TYPE="tiny"
WARMUP_STEPS=10

DATASET_TYPE="small"
WARMUP_STEPS=25

DATASET_TYPE="medium"
WARMUP_STEPS=50

DATASET_TYPE="large"
WARMUP_STEPS=100

##########
# Config #
##########
WER_THRESHOLD=10.0
TEACHER_MODEL="openai/whisper-large-v3"
HF_ORG="asahi417"
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"
huggingface-cli login

####################
# Download Dataset #
####################
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100

###################
# Generate Labels #
###################
accelerate launch scripts/run_pseudo_labelling.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${PWD}/scripts/reazonspeech_manual_dataloader.py" \
  --dataset_config_name "${DATASET_TYPE}" \
  --dataset_split_name "train" \
  --text_column_name "transcription" \
  --id_column_name "name" \
  --per_device_eval_batch_size 4 \
  --dtype "bfloat16" \
  --dataloader_num_workers 128 \
  --preprocessing_num_workers 128 \
  --logging_steps 50000 \
  --max_label_length 128 \
  --language "ja" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --overwrite_output_dir \
  --output_dir "${HF_DATASET_ALIAS}" \
  --wandb_project "wandb.${HF_DATASET_ALIAS}" \
  --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}" \
  --push_to_hub


#####################
# Filtering Dataset #
#####################
python scripts/run_data_filtering.py \
  -d "${HF_ORG}/${HF_DATASET_ALIAS}" \
  --dataset_config_name "${DATASET_TYPE}" \
  --wer_threshold ${WER_THRESHOLD} \
  --text_column_name "transcription" \
  --preprocessing_num_workers 64 \
  --max_label_length 128

############################
# Initialize Student Model #
############################
huggingface-cli repo create "${HF_MODEL_ALIAS}"
git lfs install
git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
cp scripts/create_student_model.py "${HF_MODEL_ALIAS}"
cp scripts/run_distillation.py "${HF_MODEL_ALIAS}"
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
cp ../scripts/run_distillation.py ./
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized" \
  --train_dataset_config_name "${DATASET_TYPE}" \
  --language "ja" \
  --max_label_length 128 \
  --train_split_name "train" \
  --save_steps 2500 \
  --warmup_steps "${WARMUP_STEPS}" \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 50 \
  --save_total_limit 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --preprocessing_num_workers 64 \
  --dataloader_num_workers 1 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --freeze_encoder \
  --push_to_hub \
  --do_train \
  --overwrite_output_dir \
  --num_train_epochs 8

##########################
# Evaluate Student Model #
##########################
export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut-basic5000" "asahi417/ja_asr.common_voice_8_0" "asahi417/ja_asr.reazonspeech_test"
do
  accelerate launch scripts/run_short_form_eval.py \
    --model_name_or_path "${HF_ORG}/${HF_MODEL_ALIAS}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size 512 \
    --dtype "bfloat16" \
    --dataloader_num_workers 32 \
    --preprocessing_num_workers 32 \
    --generation_max_length 256 \
    --language "ja" \
    --wandb_project "wandb.${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
    --attn_type "flash_attn"
done

#####################################
# (Optional) Evaluate Teacher Model #
#####################################
WHISPER_MODEL="openai/whisper-tiny"
BATCH_SIZE=256
WHISPER_MODEL="openai/whisper-small"
BATCH_SIZE=128
WHISPER_MODEL="openai/whisper-medium"
BATCH_SIZE=64
WHISPER_MODEL="openai/whisper-large-v3"
BATCH_SIZE=32

export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut-basic5000" "asahi417/ja_asr.common_voice_8_0" "asahi417/ja_asr.reazonspeech_test"
do
  accelerate launch scripts/run_short_form_eval.py \
    --model_name_or_path "${WHISPER_MODEL}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${WHISPER_MODEL##*/}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --dtype "bfloat16" \
    --dataloader_num_workers 32 \
    --preprocessing_num_workers 32 \
    --generation_max_length 256 \
    --language "ja" \
    --wandb_project "wandb.${WHISPER_MODEL##*/}.${EVAL_DATASET##*/}" \
    --attn_type "flash_attn"
done


####################
# Trouble Shooting #
####################
# SSL Error
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export CURL_CA_BUNDLE=''

