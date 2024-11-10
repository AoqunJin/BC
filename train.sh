export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

deepspeed --master_port 29500 train.py 

# nohup bash train.sh>output.log 2>&1 & disown
