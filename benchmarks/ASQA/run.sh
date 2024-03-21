rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
output_dir="benchmarks/ASQA/output"
cd $rageval_dir
echo "Running ASQA Benchmark"
if [ -z "$(ls -A benchmarks/ASQA/output)" ]; then
    echo "Generating ASQA examples"
    python3 $rageval_dir/benchmarks/ASQA/generate.py \
        --max_num_examples 500 \
        --max_new_tokens 256 \
        --output_dir $output_dir \
        --model "gpt-3.5-turbo-instruct" \
        --api_key "YOUR_API_KEY"
fi
echo "Running Evaluation"
python3 $rageval_dir/benchmarks/ASQA/asqa_benchmark.py --output_dir $output_dir
echo "ASQA Benchmark Complete"