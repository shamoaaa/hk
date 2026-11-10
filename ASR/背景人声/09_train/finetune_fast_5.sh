
# start
# nohup 09_train/finetune_fast.sh > finetune.out 2>&1 &
source ~/.bashrc

conda info -e

conda activate asr

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically
# model_name_or_model_dir="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_name_or_model_dir="/data1/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

## option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}


# data dir, which contains: train.json, val.json
data_dir="/data1/label/all_label"

train_text="${data_dir}/train/text"
train_scp="${data_dir}/train/wav.scp"
train_data="${data_dir}/train/train.jsonl"

dev_text="${data_dir}/dev/text"
dev_scp="${data_dir}/dev/wav.scp"
dev_data="${data_dir}/dev/dev.jsonl"

# python /data1/code/08_split/merge_all_labels.py

# generate train.jsonl and val.jsonl from wav.scp and text.txt
# python /data1/code/09_train/fast_scp2jsonl.py \
#     --wav_scp "${train_scp}" \
#     --text "${train_text}" \
#     --output "${train_data}" \
#     --workers 4 \
#     # --resume

# python /data1/code/09_train/fast_scp2jsonl.py \
#     --wav_scp "${dev_scp}" \
#     --text "${dev_text}" \
#     --output "${dev_data}" \
#     --workers 4 \
#     # --resume

# exp output dir
output_dir="/data1/train_outputs"
log_file="${output_dir}/log.txt"
mkdir -p ${output_dir}
echo "log_file: ${log_file}"

# 让 Python 优先加载本地 funasr 代码，而非 pip 安装的 site-packages 版本
export PYTHONPATH="/data1/code/09_train:${PYTHONPATH}"

torchrun \
--nnodes 1 \
--nproc_per_node ${gpu_num} \
/data1/code/09_train/funasr/bin/train.py \
++model="${model_name_or_model_dir}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${dev_data}" \
++dataset_conf.batch_size=60000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=${gpu_num} \
++train_conf.max_epoch=60 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=4000 \
++train_conf.save_checkpoint_interval=4000 \
++train_conf.avg_keep_nbest_models_type='loss' \
++train_conf.keep_nbest_models=5 \
++train_conf.avg_nbest_model=5 \
++train_conf.find_unused_parameters=true \
++freeze_param=['bias_encoder','hotword_output_layer','seaco_decoder'] \
++optim_conf.lr=0.00002 \
++output_dir="${output_dir}" \
  2>&1 | tee "${log_file}"
#   #  ['specaug','encoder','decoder','predictor']
# cp "${output_dir}/model.pt.avg10" "/home/wangsheng30/source/asr/seaco_ft_20_0.0001_freezeHw/model.pt"
# python /home/wangsheng30/release/agi_asr/test_forward.py
# python /home/wangsheng30/tools/compute-cer.py --char=1 --v=1 "/home/wangsheng30/examples/sp4_vs_sibichi/text/label" "/home/wangsheng30/examples/sp4_vs_sibichi/text/seaco_rec_ft.txt"
