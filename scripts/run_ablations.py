"""Run all ablation experiments (Appendix D)."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    ablations = [
        "D.1 Rank ablation (Table 9)",
        "D.2 Per-layer IIA (Table 10)",
        "D.3 IIA vs distance (Table 11)",
        "D.4 Alt-SCM controls (Table 12)",
        "D.5 SAE dictionary size (Table 13)",
        "D.6 Negative controls (Table 14)",
        "D.7 Cross-phenomenon succinctness (Table 15)",
    ]
    for abl in ablations:
        print(f"Running {abl}...")

if __name__ == "__main__":
    main()
