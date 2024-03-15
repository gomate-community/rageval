rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $rageval_dir
echo "Running Auto Benchmark"
python3 $rageval_dir/benchmarks/auto/auto_benchmark.py --corpus_dir "benchmarks/auto/corpus"\
                          --output_dir "benchmarks/auto/output"\
                          --model "gpt-3.5-turbo"\
                          --api_key "YOUR_API_KEY"
echo "Auto Benchmark Complete"