#!/bin/bash

# Parse command-line arguments
for i in "$@"
do
case $i in
    --input_dir=*)
    input_dir="${i#*=}"
    shift
    ;;
    --output_dir=*)
    output_dir="${i#*=}"
    shift
    ;;
    *)
    echo "Unknown option: $i"
    exit 1
    ;;
esac
done

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop over each .laz file
for file in "$input_dir"/*.laz; do
  # Create a temporary file to list the pipeline for the current file
  temp_pipeline=$(mktemp)

  # Get the bounds of the current file using pdal info and awk
  x_min=$(pdal info --metadata $file | grep minx  | grep -oP '"minx":\s*\K\d+')
  y_min=$(pdal info --metadata $file | grep miny  | grep -oP '"miny":\s*\K\d+')

  stat_file=$(basename "${file%.laz}.txt")
  stat_file=${output_dir}/${stat_file}
  echo "${x_min} ${y_min}" > ${stat_file}
  rm "$temp_pipeline"
done

echo "stat"
exit 0
