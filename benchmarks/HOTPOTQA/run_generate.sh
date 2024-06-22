rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cache_dir="$rageval_dir/HOTPOTQA"
cd $rageval_dir
echo "Generating HOTPOTQA examples"
python3 benchmarks/HOTPOTQA/generate.py \
        --subset "distractor"\
        --num_documents 10 \
        --max_num_examples 500 \
        --max_length 4096 \
        --output_path "benchmarks/HOTPOT/output" \
        --cache_path $cache_dir \
        --model "gpt-3.5-turbo" \
        --api_key "YOUR_API_KEY"