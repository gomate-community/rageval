script_dir=$(cd $(dirname $0);pwd)
cache_dir=$(dirname $(dirname $script_dir))/.rageval
wget -c https://huggingface.co/datasets/THUDM/webglm-qa/resolve/main/data/test.jsonl -O $cache_dir/datasets/webglm-test.jsonl
python3 setup.py install

#python3 $script_dir/generate.py\
#  --cache_path $cache_dir\
#  --model Llama-2-7b-chat-hf

python3 $script_dir/webglm_benchmark.py\
  --cache_path $cache_dir\
  --remote_split Llama_2_7b_chat_hf