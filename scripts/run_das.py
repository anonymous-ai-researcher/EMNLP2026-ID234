"""Run DAS verification on from-scratch models."""
import argparse, yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"Running DAS with ranks {args.ranks}")
    print(f"Config: {config['das']}")
    # See src/das/das_trainer.py for implementation

if __name__ == "__main__":
    main()
