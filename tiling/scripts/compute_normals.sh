#!/bin/bash

# Parse command-line arguments
for i in "$@"
do
  case $i in
    --project_path=*)
      PROJECT_PATH="${i#*=}"
      shift
      ;;
    --cgal_dir=*)
      CGAL_DIR="${i#*=}"
      shift
      ;;
    --output_dir=*)
      OUTPUT_DIR="${i#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $i"
      exit 1
      ;;
  esac
done

if [ -d "${PROJECT_PATH}" ]; then
    echo "processing ${PROJECT_PATH} ..." 
else
    echo "${PROJECT_PATH} does not exists! create it" 
    exit 1
fi

export PROJECT_PATH
output_dir=${OUTPUT_DIR}
mkdir -p ${output_dir}
process_file() {
  input_file=$1
  output_file="${output_dir}/$(basename "$file" .laz).ply"
  docker compose --project-directory $CGAL_DIR run -T  --rm cgal /bin/bash -c "/usr/src/app/cgal/build/compute_normals ${input_file} ${output_file}"
}

total_cores=$(nproc)
max_jobs=8 #$((total_cores - 1))

count=0


for file in "${PROJECT_PATH}"/merged/*.laz; do
  process_file "$file" &
  count=$((count + 1))

  # Attendre si le nombre maximum de tâches est atteint
  if [[ $count -ge $max_jobs ]]; then
    wait -n
    count=$((count - 1))
  fi
done


