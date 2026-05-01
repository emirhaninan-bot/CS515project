from parameters import get_args
from train import run_training_loop, train_late_fusion
from test import evaluate_test_set


def main():
    """Parse CLI arguments and dispatch to the selected execution mode.

    Modes
    -----
    train  : Train the PerturbationGAT model from scratch (two-phase curriculum).
    fusion : Train the LateFusionGNN head on top of a frozen PerturbationGAT expert.
    test   : Evaluate the LateFusionGNN on the held-out gene-level test split.

    Example usage
    -------------
    python main.py train --batch-size 32 --phase2-epochs 50 --hidden-dim 256
    python main.py fusion --fusion-epochs 20 --data-dir /path/to/data
    python main.py test
    """
    args = get_args()

    if args.mode == "train":
        run_training_loop(args)
    elif args.mode == "fusion":
        train_late_fusion(args)
    elif args.mode == "test":
        evaluate_test_set(args)


if __name__ == "__main__":
    main()
