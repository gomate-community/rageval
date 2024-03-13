import argparse

from eli5 import ELI5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")  # openai model/local LLM
    parser.add_argument("--api_key", type=str, default=None)  # api_key is necessary if using openai model
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="bm25")  # bm25/oracle
    parser.add_argument("--method", type=str, default="vanilla")
    parser.add_argument("--ndoc", type=int, default=5)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--metrics", nargs="+", type=str, default=["nli_claim"])
    args = parser.parse_args()

    eli5 = ELI5(args)
    eli5.init_model(args.model, args.api_key)
    gen_result = eli5.predict()
    eli5.evaluate()
    eli5.save_result()
