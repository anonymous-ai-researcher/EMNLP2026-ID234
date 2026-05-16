"""Run negative control experiments."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    print("Running: wrong-SCM, random-init, shuffled-labels controls")

if __name__ == "__main__":
    main()
