rageval_dir=$(dirname $(dirname "$(dirname "$0")"))
cd "$rageval_dir"
python3 $rageval_dir/benchmark/auto/auto_benchmark.py --corpus_dir "benchmark/auto/corpus"\
                          --output_dir "benchmark/auto/output"\
                          --model "gpt-3.5-turbo"\
                          --api_key "YOUR_API_KEY"