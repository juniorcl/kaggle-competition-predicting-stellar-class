import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Code for training estimators")

    parser.add_argument("--layer", type="str", help="Training models for the layer one")

    parser.parse_args()

    if parser.layer == "one":
