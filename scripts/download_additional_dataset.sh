#!/bin/bash

hf download --repo-type dataset HNO333333/TSG additional_competition_data.zip --local-dir ./

unzip additional_competition_data.zip

find ./raid -type f -name "*.zip" | while read zipfile; do
    # Extract <comp> from the path
    comp=$(basename "$zipfile" .zip)

    # Create destination directory if it doesn't exist
    mkdir -p "~/.cache/TimeSeriesGym/data/$comp"

    # Copy the zip file to .data/<comp>/
    cp "$zipfile" "~/.cache/TimeSeriesGym/data/$comp"
done

rm -rf ./raid