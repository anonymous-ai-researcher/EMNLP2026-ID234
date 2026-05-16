"""Run DAS verification on pre-trained models (Pythia, Llama)."""
import argparse, yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_inits", type=int, default=3)
    args = parser.parse_args()
    
    print(f"Running DAS on {args.model_name} ({args.n_inits} inits)")

if __name__ == "__main__":
    main()
