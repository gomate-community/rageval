script_dir=$(cd $(dirname $0);pwd)
cache_dir=$(dirname $(dirname $(dirname $script_dir)))/.rageval
wget -cP $cache_dir/datasets https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar
tar -xvf $cache_dir/datasets/ALCE-data.tar -C $cache_dir/datasets
python3 setup.py install

#python3 $script_dir/generate.py\
#  --cache_path $cache_dir\
#  --model Llama-2-7b-chat-hf\
#  --dataset gtr\
#  --ndoc 5\
#  --shot 2

python3 $script_dir/asqa_benchmark.py\
  --cache_path $cache_dir\
  --remote_split Llama_2_7b_chat_hf_vanilla_shot2_ndoc5
