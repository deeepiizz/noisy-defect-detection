"""Run the noise-robustness experiment and write results to results/.

Examples:
    python main.py                                   # synthetic data, no download
    python main.py --data mvtec --root data/mvtec --category screw
"""
import argparse
import csv
import json
from pathlib import Path

from ndd import data, denoise, evaluate, noise, visualize
from ndd.detectors import DETECTORS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "mvtec"], default="synthetic")
    p.add_argument("--root", default="data/mvtec", help="MVTec AD root directory")
    p.add_argument("--category", default="screw")
    p.add_argument("--size", type=int, default=128, help="resize images to size x size")
    p.add_argument("--noise", choices=list(noise.NOISE), default="gaussian")
    p.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--denoisers", nargs="+", default=["none", "median", "wiener"], choices=list(denoise.FILTERS))
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = data.make_synthetic(size=args.size) if args.data == "synthetic" else data.load_mvtec(args.root, args.category, args.size)
    print(f"train={len(split.train)}  test={len(split.test)}  defective={split.labels.sum()}")

    add_noise = noise.NOISE[args.noise]
    rows = []
    for level in args.levels:
        # Noise hits both train and test, as it would with a noisy camera.
        train_noisy = add_noise(split.train, level, seed=1)
        test_noisy = add_noise(split.test, level, seed=2)

        for den in args.denoisers:
            train_d = denoise.apply(train_noisy, den)
            test_d = denoise.apply(test_noisy, den)
            psnr, ssim = denoise.quality(split.test, test_d)

            for det_name in args.detectors:
                det = DETECTORS[det_name]().fit(train_d)
                maps = det.anomaly_map(test_d)
                row = {
                    "noise": args.noise,
                    "noise_level": level,
                    "denoise": den,
                    "detector": det_name,
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 3),
                    "image_auroc": round(evaluate.image_auroc(split.labels, maps), 3),
                    "pixel_auroc": round(evaluate.pixel_auroc(split.masks, maps), 3),
                }
                rows.append(row)
                print("  ".join(f"{k}={v}" for k, v in row.items()))

                # Save example heatmaps at a moderate noise level for the README.
                if level == args.levels[min(1, len(args.levels) - 1)] and den == "median":
                    visualize.plot_examples(test_noisy, split.masks, maps, split.labels, out / f"examples_{det_name}.png")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    visualize.plot_robustness(rows, out / "robustness_image_auroc.png", "image_auroc")
    visualize.plot_robustness(rows, out / "robustness_pixel_auroc.png", "pixel_auroc")
    print(f"\nWrote results to {out}/")


if __name__ == "__main__":
    main()
