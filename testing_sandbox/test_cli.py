import argparse

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process input arguments for a simulation.')

    # Adding arguments

    parser.add_argument('--tags', nargs='+', default = ["experiment", "ppo"], help='Tags for wandb runs')

    # Parse the arguments
    args = parser.parse_args()

    # Print the inputs (You can replace this section with the actual logic)
    print(*args.tags)
