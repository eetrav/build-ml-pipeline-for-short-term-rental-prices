#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning.

Export the result to a new wandb artifact.

Author: Emily Travinsky
Date: 10/2023
"""
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args: dict):
    """
    Function to clean data and export new artifact.

    Args:
        args (dict): Input arguments for processing, defined in parser.
    """

    logger.info("Starting wandb run for data clean-up.")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact = run.use_artifact(args.input_artifact)

    rental_df = pd.read_csv(artifact.file())

    logger.info("Starting data clean-up.")

    # Filter dataframe to min and max nightly prices
    idx = rental_df['price'].between(args.min_price, args.max_price)
    rental_df = rental_df[idx].copy()
    # Convert last_review to datetime
    rental_df['last_review'] = pd.to_datetime(rental_df['last_review'])

    #Filter daraframe to lat/long boundaries of NYC
    idx = rental_df['longitude'].between(-74.25, -73.50) & rental_df['latitude'].between(40.5, 41.2)
    rental_df = rental_df[idx].copy()

    rental_df.to_csv(args.output_artifact, index=False)

    logger.info("Creating artifact.")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)

    logger.info("Logging artifact.")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type= str,
        help= "Fully-qualified name for the input artifact.",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type= str,
        help= "Name for the W&B artifact that will be created.",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type= str,
        help= "Type of the artifact to create.",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type= str,
        help= "Description for the artifact.",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type= float,
        help= "Minimum price for nightly rental filtering",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type= float,
        help= "Maximum price for nightly rental filtering",
        required=True
    )

    input_args = parser.parse_args()

    go(input_args)
