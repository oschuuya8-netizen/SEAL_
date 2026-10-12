#!/usr/bin/env python3
from config import configure_environment, parse_args

configure_environment()

from data import load_samples
from pipeline import SEALPipeline


def main():
    args = parse_args()
    samples = load_samples(
        args.data,
        args.limit,
        args.seed,
        args.min_response_words,
        preserve_order=args.preserve_order,
    )
    if not samples:
        raise ValueError(f"No usable samples found in {args.data}")
    payload = SEALPipeline(args).run(samples)
    print(f"Wrote {payload['summary']['num_results']} final responses to {args.output}")


if __name__ == "__main__":
    main()
