"""Grid-64 DeepRV training and inference on the 128x128 experiment."""

import argparse

from target128_common import Config, PUBLIC_SEEDS, run_nuts, run_training

MODELS = ('Bilinear64', 'Cubic64', 'DTC64', 'FITC64')
FORMAL_INFERENCE_MODELS = ('Bilinear64', 'Cubic64', 'DTC64')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'infer', 'all'), default='all')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=None)
    parser.add_argument('--output-root', default='outputs/target128')
    args = parser.parse_args()
    selected_models = tuple(args.models) if args.models is not None else MODELS
    cfg = Config(output_root=args.output_root)
    if args.stage in ('train', 'all'):
        for model in selected_models:
            run_training(cfg, model, probe=False)
    if args.stage in ('infer', 'all'):
        if args.models is not None:
            ineligible = set(selected_models).difference(FORMAL_INFERENCE_MODELS)
            if ineligible:
                raise RuntimeError(
                    'Formal inference refused because these models have no '
                    f'eligible formal checkpoint: {sorted(ineligible)}'
                )
        inference_models = (
            selected_models
            if args.models is not None
            else FORMAL_INFERENCE_MODELS
        )
        for model in inference_models:
            for seed in PUBLIC_SEEDS:
                run_nuts(cfg, model, seed, probe=False)

if __name__ == '__main__':
    main()
