"""Run SAE analysis on from-scratch Transformer."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()
    
    print(f"Training SAE: expansion={args.expansion}, K={args.topk}, steps={args.steps}")

if __name__ == "__main__":
    main()
