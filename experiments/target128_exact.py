"""Exact DeepRV training and inference on the 128x128 experiment."""

import argparse

from target128_common import Config, PUBLIC_SEEDS, run_nuts, run_training

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'infer', 'all'), default='all')
    parser.add_argument('--output-root', default='outputs/target128')
    args = parser.parse_args()
    cfg = Config(output_root=args.output_root)
    if args.stage in ('train', 'all'):
        run_training(cfg, 'Exact128', probe=False)
    if args.stage in ('infer', 'all'):
        for seed in PUBLIC_SEEDS:
            run_nuts(cfg, 'Exact128', seed, probe=False)

if __name__ == '__main__':
    main()
