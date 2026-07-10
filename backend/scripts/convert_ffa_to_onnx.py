import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.ffa_net import convert_ffa_to_onnx


def main():
    parser = argparse.ArgumentParser(description='Convert FFA-Net PyTorch model to ONNX')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to FFA-Net PyTorch model (.pk file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX model path')
    parser.add_argument('--gps', type=int, default=3,
                        help='Number of groups in FFA (default: 3)')
    parser.add_argument('--blocks', type=int, default=19,
                        help='Number of blocks per group (default: 19)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "weights", "enhancement", f"{base_name}.onnx"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Converting FFA-Net model...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  GPS: {args.gps}, Blocks: {args.blocks}")

    convert_ffa_to_onnx(args.input, args.output, gps=args.gps, blocks=args.blocks)
    print("Conversion completed!")


if __name__ == "__main__":
    main()
