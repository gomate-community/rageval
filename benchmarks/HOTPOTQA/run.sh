rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $rageval_dir

echo "Running HotPotQA Benchmark"

python3 benchmarks/HOTPOTQA/hotpot_qa_benchmark.py --output_dir "benchmarks/HOTPOTQA" --remote_split "gpt_3.5_turbo"

# Check return status code
if [ $? -eq 0 ]; then
    echo "HotPotQA Benchmark Complete"
else
    echo "Error: HotPotQA Benchmark failed to execute."
fi

