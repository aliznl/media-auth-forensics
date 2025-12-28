import argparse
import json
from media_auth_forensics.infer.pipeline import infer_path, adversarial_folder_test


def main() -> None:
    """
    Entry point for the command-line interface.
    Provides:
      - infer: analyze one image or video and output JSON report
      - adversarial_test: run adversarial perturbations on a folder of images
    """
    parser = argparse.ArgumentParser(prog="media-auth-forensics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="Run inference on an image or video.")
    p_infer.add_argument("--input", required=True, help="Path to image/video.")
    p_infer.add_argument("--checkpoint", default=None, help="Path to detection model checkpoint.")
    p_infer.add_argument("--id_checkpoint", default=None, help="Optional model-ID checkpoint.")
    p_infer.add_argument("--device", default="cpu", help="cpu|cuda")
    p_infer.add_argument("--out", default="report.json", help="Output JSON path.")
    p_infer.add_argument("--max_frames", type=int, default=120, help="Max frames for video sampling.")
    p_infer.add_argument("--stride", type=int, default=5, help="Frame stride for videos.")
    p_infer.add_argument("--no_adversarial", action="store_true", help="Disable adversarial variants during infer.")

    p_adv = sub.add_parser("adversarial_test", help="Run adversarial suite on a folder of images.")
    p_adv.add_argument("--input_dir", required=True, help="Folder with images.")
    p_adv.add_argument("--output_dir", required=True, help="Where to write JSON results.")

    args = parser.parse_args()

    if args.cmd == "infer":
        report = infer_path(
            input_path=args.input,
            detection_checkpoint=args.checkpoint,
            id_checkpoint=args.id_checkpoint,
            device=args.device,
            max_frames=args.max_frames,
            stride=args.stride,
            run_adversarial_variants=not args.no_adversarial,
        )
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(args.out)

    elif args.cmd == "adversarial_test":
        adversarial_folder_test(args.input_dir, args.output_dir)
        print(args.output_dir)


if __name__ == "__main__":
    main()
