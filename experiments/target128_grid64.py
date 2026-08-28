"""Grid-64 DeepRV training and inference on the 128x128 experiment."""

import argparse

from target128_common import Config, PUBLIC_SEEDS, run_nuts, run_training

MODELS = ('Bilinear64', 'Cubic64', 'DTC64', 'FITC64')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'infer', 'all'), default='all')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--output-root', default='outputs/target128')
    args = parser.parse_args()
    cfg = Config(output_root=args.output_root)
    if args.stage in ('train', 'all'):
        for model in args.models:
            run_training(cfg, model, probe=False)
    if args.stage in ('infer', 'all'):
        for model in args.models:
            for seed in PUBLIC_SEEDS:
                run_nuts(cfg, model, seed, probe=False)

if __name__ == '__main__':
    main()
