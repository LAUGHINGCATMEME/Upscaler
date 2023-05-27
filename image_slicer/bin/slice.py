#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will be deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
from image_slicer import slice, save_tiles, get_basename


def main():
    """Parse arguments and slice image."""
    parser = construct_parser()
    args = parser.parse_args()
    if args.num_tiles == 0 and args.rows == 1 and args.columns == 1:
        parser.error(
            "No operation specified. You need to either specify the"
            "number of tiles to slice automatically, or specify the"
            "row and columns to customize the slice."
        )
    tiles = slice(
        args.image,
        number_tiles=args.num_tiles,
        row=args.rows,
        col=args.columns,
        save=False,
    )
    save_tiles(
        tiles, prefix=get_basename(args.image), directory=args.dir, format=args.format
    )


def construct_parser():
    """Return an ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="slice-image",
        description="Slice an image into tiles.",
        epilog="Report bugs and make feature requests at"
        "https://github.com/samdobson/image_slicer/issues",
        add_help=False,
    )

    required = parser.add_argument_group("Required Arguments")
    required.add_argument("image", help="image file")

    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument(
        "-n",
        "--num-tiles",
        type=int,
        default=0,
        help="Number of tiles to make. Automatically decides the"
        "number of rows and columns.",
    )
    optional.add_argument("-d", "--dir", default="./", help="output directory")
    optional.add_argument(
        "-f", "--format", default="png", help="output image format (e.g JPEG, PNG, GIF)"
    )
    optional.add_argument(
        "-r",
        "--rows",
        type=int,
        default=1,
        help="Number of rows to divide the image. Used when num_tiles is 0.",
    )
    optional.add_argument(
        "-c",
        "--columns",
        type=int,
        default=1,
        help="Number of columns to divide the image. Used when num_tiles is 0.",
    )

    info = parser.add_argument_group("Info")
    info.add_argument("-h", "--help", action="help", help="display this screen"),
    info.add_argument("-v", "--version", action="version", version="%(prog)s 0.2")

    return parser


if __name__ == "__main__":
    main()
