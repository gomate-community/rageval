rageval_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $rageval_dir
echo "Running ASQA Benchmark"
python3 benchmarks/ASQA/asqa_benchmark.py --output_dir ".rageval/benchmark" --split "gpt_3.5_turbo_instruct"
echo "ASQA Benchmark Complete"