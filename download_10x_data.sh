#!/bin/bash

# Check if "data" folder already exists
if [ -d "data" ]; then
  echo "⚠️  'data' directory already exists. Aborting to avoid overwriting."
  exit 1
fi

# Create base data directory
mkdir -p "data"

# Define all the sample names
samples=(
  "Visium_HD_Human_Colon_Cancer_P1"
  "Visium_HD_Human_Colon_Cancer_P2"
  "Visium_HD_Human_Colon_Cancer_P5"
  "Visium_HD_Human_Colon_Normal_P3"
  "Visium_HD_Human_Colon_Normal_P5"
)

# Define file suffixes for input and output
declare -A input_suffixes
declare -A output_suffixes

input_suffixes["Cancer"]="probe_set.csv image.tif alignment_file.json tissue_image.btf"
output_suffixes["Cancer"]="binned_outputs.tar.gz molecule_info.h5 feature_slice.h5"

input_suffixes["Normal"]="probe_set.csv image.tif alignment_file.json tissue_image.btf"
output_suffixes["Normal"]="binned_outputs.tar.gz molecule_info.h5 feature_slice.h5"

# Base URL for standard files
base_url="https://cf.10xgenomics.com/samples/spatial-exp/3.0.0"

# Base URL for parquet metadata (GitHub raw access)
parquet_base_url="https://github.com/10XGenomics/HumanColonCancer_VisiumHD/raw/main/MetaData"

for sample in "${samples[@]}"; do
  if [[ $sample == *"Cancer"* ]]; then
    type="CRC"
    input_list=${input_suffixes["Cancer"]}
    output_list=${output_suffixes["Cancer"]}
  elif [[ $sample == *"Normal"* ]]; then
    type="Normal"
    input_list=${input_suffixes["Normal"]}
    output_list=${output_suffixes["Normal"]}
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

  # Set up directory structure
  dir="data/${type}_${person}"
  input_dir="$dir/input"
  output_dir="$dir/output"
  mkdir -p "$input_dir" "$output_dir"

  # Download input files
  for suffix in $input_list; do
    filename="${sample}_${suffix}"
    url="${base_url}/${sample}/${filename}"
    echo "⬇️  Downloading $filename → $input_dir/"
    curl --retry 3 --retry-delay 5 --connect-timeout 10 -f -s -O "$url" && mv "$filename" "$input_dir/" || echo "❌ Failed to download $filename"
  done

  # Download output files
  for suffix in $output_list; do
    filename="${sample}_${suffix}"
    url="${base_url}/${sample}/${filename}"
    echo "⬇️  Downloading $filename → $output_dir/"
    curl --retry 3 --retry-delay 5 --connect-timeout 10 -f -s -O "$url" && mv "$filename" "$output_dir/" || echo "❌ Failed to download $filename"
  done

  # Download Metadata.parquet file for cancer samples
  if [[ $type == "CRC" ]]; then
    parquet_file="${person}CRC_Metadata.parquet"
    parquet_url="${parquet_base_url}/${parquet_file}?raw=true"
    echo "⬇️  Downloading metadata $parquet_file → $dir/"

    curl --retry 3 --retry-delay 5 --connect-timeout 10 -f -S -L -o "$dir/$parquet_file" "$parquet_url" || {
      echo "❌ Failed to download $parquet_file"
      continue
    }

    # Check if file is non-empty
    if [[ ! -s "$dir/$parquet_file" ]]; then
      echo "⚠️  Downloaded file $parquet_file is empty. Deleting."
      rm "$dir/$parquet_file"
    fi
  fi
done

echo "✅ All downloads completed."

