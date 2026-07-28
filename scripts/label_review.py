from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch the TrackGuard annotation review tool")
    parser.add_argument("--dir", default=None,
                        help="Sequence directory containing gt/gt.txt and img1/")
    parser.add_argument("--no_auto_save", action="store_true")
    args = parser.parse_args()

    from labeling.app import launch
    launch(directory=args.dir, auto_save=not args.no_auto_save)


if __name__ == "__main__":
    main()
