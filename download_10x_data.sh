#!/bin/bash

# Define all the sample names
samples=(
  "Visium_HD_Human_Colon_Cancer_P1"
  "Visium_HD_Human_Colon_Cancer_P2"
  "Visium_HD_Human_Colon_Cancer_P5"
  "Visium_HD_Human_Colon_Normal_P3"
  "Visium_HD_Human_Colon_Normal_P5"
)

# Define file suffixes to download
declare -A suffixes
suffixes["Cancer"]="probe_set.csv image.tif alignment_file.json molecule_info.h5 spatial.tar.gz feature_slice.h5"
suffixes["Normal"]="probe_set.csv image.tif alignment_file.json molecule_info.h5 spatial.tar.gz feature_slice.h5"

# Base URL for standard files
base_url="https://cf.10xgenomics.com/samples/spatial-exp/3.0.0"

# Base URL for parquet metadata (GitHub raw access)
parquet_base_url="https://github.com/10XGenomics/HumanColonCancer_VisiumHD/raw/main/MetaData"

for sample in "${samples[@]}"; do
  if [[ $sample == *"Cancer"* ]]; then
    type="CRC"
    suffix_list=${suffixes["Cancer"]}
  elif [[ $sample == *"Normal"* ]]; then
    type="Normal"
    suffix_list=${suffixes["Normal"]}
  else
    echo "Unknown sample type: $sample"
    continue
  fi

  # Extract person ID (e.g., P1, P2, etc.)
  if [[ $sample =~ (P[0-9]+) ]]; then
    person=${BASH_REMATCH[1]}
  else
    echo "Failed to extract person ID from $sample"
    continue
  fi

  # Create directory
  dir="${type}_${person}"
  mkdir -p "$dir"

  # Download and rename files
  for suffix in $suffix_list; do
    filename="${sample}_${suffix}"
    url="${base_url}/${sample}/${filename}"
    output_filename="$(basename "$suffix")"
    echo "Downloading $filename → $dir/$output_filename"
    curl --retry 3 --retry-delay 5 --connect-timeout 10 -f -s -O "$url" && mv "$filename" "$dir/$output_filename" || echo "❌ Failed to download $filename"
  done

  # Download additional metadata parquet file for cancer samples
  if [[ $type == "CRC" ]]; then
    parquet_file="${person}CRC_Metadata.parquet"
    parquet_url="${parquet_base_url}/${parquet_file}?raw=true"
    echo "Downloading metadata $parquet_file → $dir/"

    curl --retry 3 --retry-delay 5 --connect-timeout 10 -f -S -L -o "$dir/$parquet_file" "$parquet_url" || {
      echo "Failed to download $parquet_file"
      continue
    }

    # Check if file is non-empty
    if [[ ! -s "$dir/$parquet_file" ]]; then
      echo "Downloaded file $parquet_file is empty. Deleting."
      rm "$dir/$parquet_file"
    fi
  fi
done

echo "All downloads completed."
