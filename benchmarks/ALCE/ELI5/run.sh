script_dir=$(cd $(dirname $0);pwd)
cache_dir=$(dirname $(dirname $(dirname $script_dir)))/.rageval
wget -cP $cache_dir/datasets https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar
tar -xvf $cache_dir/datasets/ALCE-data.tar -C $cache_dir/datasets
python3 setup.py install
python3 $script_dir/run.py\
  --cache_path $cache_dir\
  --model $cache_dir/models/Llama-2-7b-chat-hf\
  --dataset bm25\
  --method vanilla\
  --ndoc 5\
  --shot 2\
  --metrics nli_claim citation_recall citation_precision\
