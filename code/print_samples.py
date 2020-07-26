import util
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    
    args = parser.parse_args()

    texts = util.read_file(args.file)
    sampled_texts = util.get_samples(texts, n_samples=20)

    print("Sampled texts")
    print("--------------------")
    for t in sampled_texts:
        print(t)
