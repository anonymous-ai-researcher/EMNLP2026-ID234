"""Train from-scratch models (Transformer, LSTM, Mamba) with grokking protocol."""
import argparse, yaml, torch, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phenomenon", type=str, default="agreement")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stage", type=str, default="train", choices=["data", "train"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.stage == "data":
        from src.data.cfg_generator import CFGGenerator
        for phen in ["agreement", "npi", "binding", "concord"]:
            gen = CFGGenerator(phen, seed=config["data"]["seed"])
            data = gen.generate_dataset(n_per_distance=config["data"]["train_size"] // 9)
            os.makedirs(f"data/{phen}", exist_ok=True)
            torch.save(data, f"data/{phen}/train.pt")
            print(f"Generated {len(data)} examples for {phen}")
    else:
        print(f"Training {config['model']['type']} on {args.phenomenon} (seed {args.seed})")
        # Model training loop (see src/models/ for architecture details)
        pass

if __name__ == "__main__":
    main()
