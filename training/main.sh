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

