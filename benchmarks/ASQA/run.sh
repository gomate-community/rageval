rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $rageval_dir
echo "Running ASQA Benchmark"
python3 benchmarks/ASQA/asqa_benchmark.py --output_dir ".rageval/results" --dataset_path "benchmarks/ASQA/data/gpt-3.5-turbo-instruct.jsonl"
echo "ASQA Benchmark Complete"