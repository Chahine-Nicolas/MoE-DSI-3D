#!/bin/bash

# Bounding box size (100 meters)
bbox_size=500
subsample_ratio=0.5
max_parallel_jobs=16  # Set maximum parallel processes

# Parse command-line arguments
for i in "$@"; do
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
        --LAT=*)
            LAT="${i#*=}"
            shift
            ;;
        --LONG=*)
            LONG="${i#*=}"
            shift
            ;;
	--subsample_ratio=*)
	    subsample_ratio="${i#*=}"
	    shift
	    ;;    	
        --max_jobs=*)
            max_parallel_jobs="${i#*=}"
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

# Create a temporary directory for sampled files
temp_dir="${input_dir}/sampled/"
mkdir -p ${temp_dir} 


# Function to control parallel processes
run_with_limit() {
    while [ "$(jobs | wc -l)" -ge "$max_parallel_jobs" ]; do
        wait -n  # Wait for any job to complete before starting a new one
    done
}

# Sample each file in parallel with limited jobs
for file in "$input_dir"/*.laz; do
    sampled_file="$temp_dir/$(basename "$file")"

    # Ensure we only launch a new job if we are below the max_parallel_jobs limit
    run_with_limit
    
    pdal pipeline -i <(
        cat <<EOF
        {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": "$file"
                },
                {
                    "type": "filters.sample",
                    "radius": "$subsample_ratio"
                },
                {
                    "type": "writers.las",
                    "filename": "$sampled_file",
                    "compression": "laszip"
                }
            ]
        }
EOF
    ) &  # Run in the background
done

# Wait for all background jobs to complete
wait


echo "Cropping completed. Result saved to $output_file"
exit 0
