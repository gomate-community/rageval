rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $rageval_dir
echo "Generating ASQA examples"
python3 benchmarks/ASQA/generate.py \
        --max_num_examples 500 \
        --max_new_tokens 256 \
        --output_path "benchmarks/ASQA/output" \
        --model "gpt-3.5-turbo-instruct" \
        --api_key "YOUR_API_KEY"