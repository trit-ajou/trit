import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--no_grad", action="store_true")
    parser.add_argument("--continue", action="store_true")
    parser.add_argument("--use_noise", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--ckpt_interval", type=int, default=5)

    args = parser.parse_args()

    if args.model == 0:
        args.no_grad = True

    return args


if __name__ == "__main__":
    args = parse_args()
