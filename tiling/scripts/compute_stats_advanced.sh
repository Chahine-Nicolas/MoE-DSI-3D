#!/bin/bash

# Parse command-line arguments
for i in "$@"; do
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

# Créer le répertoire de sortie s'il n'existe pas
mkdir -p "$output_dir"

# Fonction de traitement pour chaque fichier
process_file() {
  file="$1"
  echo "file => $file"
  filename=$(basename "$file")
  base="${filename%.*}"
  output_json="${output_dir}/${base}.stats.json"

  pdal info  $file > "$output_json"

}

export -f process_file
export output_dir

# Nombre de cœurs CPU pour le parallélisme
total_cores=$(nproc)
max_jobs=8 #$((total_cores - 1))
count=0

# Boucle sur chaque fichier .laz
for file in "$input_dir"/*.laz; do
  process_file "$file" &
  count=$((count + 1))

  # Si on atteint le nombre max de jobs en parallèle, on attend
  if [[ $count -ge $max_jobs ]]; then
    wait -n
    count=$((count - 1))
  fi
done

# Attendre la fin de toutes les tâches restantes
wait

exit 0
