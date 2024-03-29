
##########
# Config #
##########
DATASET_TYPE="all"
WARMUP_STEPS=500
WER_THRESHOLD=10.0
TEACHER_MODEL="openai/whisper-large-v3"
HF_ORG="asahi417"
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"
huggingface-cli login

process_chunk () {
  DATASET_CHUNK_ID=${1}
  CHUNK_START=${2}
  CHUNK_END=${3}
  export WANDB_DISABLED="true"
  export PREPROCESSING_ONLY=0
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  accelerate launch --multi_gpu scripts/run_pseudo_labelling.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --dataset_name "${PWD}/scripts/reazonspeech_manual_dataloader.py" \
    --dataset_config_name "${DATASET_TYPE}" \
    --dataset_dir_suffix "${CHUNK_START}_${CHUNK_END}" \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 1 \
    --logging_steps 100 \
    --max_label_length 128 \
    --language "ja" \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
    --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"
}
process_chunk () {
  DATASET_CHUNK_ID=${1}
  CHUNK_START=${2}
  CHUNK_END=${3}
  python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s "${CHUNK_START}" -e "${CHUNK_END}"
}

####################
# Download Dataset #
####################
process_chunk 1 0 50
process_chunk 2 50 100
process_chunk 3 100 150
process_chunk 4 150 200
# 1
DATASET_CHUNK_ID=1
CHUNK_START=0
CHUNK_END=400
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=2
CHUNK_START=400
CHUNK_END=800
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

# 2
DATASET_CHUNK_ID=3
CHUNK_START=800
CHUNK_END=1200
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=4
CHUNK_START=1200
CHUNK_END=1600
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=5
CHUNK_START=1600
CHUNK_END=2000
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=6
CHUNK_START=2000
CHUNK_END=2400
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=7
CHUNK_START=2400
CHUNK_END=2800
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=8
CHUNK_START=2800
CHUNK_END=3200
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=9
CHUNK_START=3200
CHUNK_END=3600
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

DATASET_CHUNK_ID=10
CHUNK_START=3600
CHUNK_END=4095  # remove the last one for eval
python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100 -s ${CHUNK_START} -e ${CHUNK_END}

###################
# Generate Labels #
###################
export WANDB_DISABLED="true"
export PREPROCESSING_ONLY=1
export CUDA_VISIBLE_DEVICES=
accelerate launch scripts/run_pseudo_labelling.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${PWD}/scripts/reazonspeech_manual_dataloader.py" \
  --dataset_config_name "${DATASET_TYPE}" \
  --dataset_dir_suffix "${CHUNK_START}_${CHUNK_END}" \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 8 \
  --logging_steps 100 \
  --max_label_length 128 \
  --language "ja" \
  --generation_num_beams 1 \
  --overwrite_output_dir \
  --output_dir "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
  --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"

export PREPROCESSING_ONLY=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --multi_gpu scripts/run_pseudo_labelling.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${PWD}/scripts/reazonspeech_manual_dataloader.py" \
  --dataset_config_name "${DATASET_TYPE}" \
  --dataset_dir_suffix "${CHUNK_START}_${CHUNK_END}" \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 16 \
  --logging_steps 100 \
  --max_label_length 128 \
  --language "ja" \
  --generation_num_beams 1 \
  --overwrite_output_dir \
  --output_dir "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
  --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"

#####################
# Filtering Dataset #
#####################
python scripts/run_data_filtering.py \
  -d "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
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

# Single Step: Log-Mel feature and distillation in a single process
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}" \
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
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 32 \
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
cd ../


# Two Steps: First, generating Log-Mel feature and save it on HF. Second, distillation where loading the Log-Mel
# feature from HF.
# - Step 1
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}" \
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
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 32 \
  --dataloader_num_workers 1 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --freeze_encoder \
  --push_to_hub \
  --do_train \
  --overwrite_output_dir \
  --logmel_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized" \
  --num_train_epochs 8

# - Step 2:
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
  --skip_logmel_transformation \
  --num_train_epochs 8

##########################
# Evaluate Student Model #
##########################
export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut-basic5000" "asahi417/ja_asr.common_voice_8_0"
do
  accelerate launch scripts/run_short_form_eval.py \
    --model_name_or_path "${HF_ORG}/${HF_MODEL_ALIAS}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size 512 \
    --dtype "bfloat16" \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 1 \
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
for EVAL_DATASET in "asahi417/ja_asr.jsut-basic5000" "asahi417/ja_asr.common_voice_8_0"
do
  accelerate launch run_short_form_eval.py \
    --model_name_or_path "${WHISPER_MODEL}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${WHISPER_MODEL##*/}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --dtype "bfloat16" \
    --dataloader_num_workers 64 \
    --preprocessing_num_workers 128 \
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

