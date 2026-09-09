#!/bin/bash


# Bounding box size (100 meters)
bbox_size=750
subsample_ratio=0.2
# Parse command-line arguments
for i in "$@"
do
case $i in
    --input_dir=*)
    input_dir="${i#*=}"
    shift
    ;;
    --output_file=*)
    output_file="${i#*=}"
    shift
    ;;
    --bbox_size=*)
    bbox_size="${i#*=}"
    shift
    ;;
    --subsample_ratio=*)
    subsample_ratio="${i#*=}"
    shift
    ;;    
    --LAT=*)
    LAT="${i#*=}"
    shift
    ;;
    --LONG=*)
    LONG="${i#*=}"
    shift
    ;;
    *)
    echo "Unknown option: $i"
    exit 1
    ;;
esac
done

# Transform the GPS coordinates (WGS84) to Lambert 93 using cs2cs
read -r LAMBERT_X LAMBERT_Y <<EOF
$(echo "$LONG $LAT" | cs2cs +init=epsg:4326 +to +init=epsg:2154 | awk '{print $1, $2}')
EOF



# Define the bounding box around the transformed Lambert 93 point
x_min=$(python3 -c "print($LAMBERT_X - $bbox_size)")
x_max=$(python3 -c "print($LAMBERT_X + $bbox_size)")
y_min=$(python3 -c "print($LAMBERT_Y - $bbox_size)")
y_max=$(python3 -c "print($LAMBERT_Y + $bbox_size)")
echo "${x_min} ${x_max} ${y_min} ${y_max}"
# Create a temporary file to list all .laz files
temp_pipeline=$(mktemp)

# Start creating the PDAL pipeline
echo '{
  "pipeline": [' > "$temp_pipeline"

# Add each .laz file to the pipeline as a separate readers.las stage
for file in "$input_dir"/*.laz; do
  echo '    {
      "type": "readers.las",
      "filename": "'"$file"'"
    },' >> "$temp_pipeline"
done

# Merge the files and apply the cropping
cat >> "$temp_pipeline" <<EOF
    {
      "type": "filters.merge"
    },
    {
      "type": "filters.crop",
      "bounds": "([$x_min, $x_max], [$y_min, $y_max])"
    },
    {
      "type": "writers.las",
      "filename": "$output_file",
      "compression": "laszip"
    }
  ]
}
EOF

# Run the PDAL pipeline using the generated file
pdal pipeline "$temp_pipeline"

# Remove the temporary pipeline file
rm "$temp_pipeline"

echo "Cropping completed. Result saved to $output_file"
exit 0
