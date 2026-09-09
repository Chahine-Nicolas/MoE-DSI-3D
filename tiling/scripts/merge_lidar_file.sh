#!/bin/bash

# Default power of 2 for tiling
pow=3
subsample_ratio=0.2

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


subtract_4digit() {
    local a="$1"
    local b="$2"

    # Convert to base-10 to avoid octal issues
    local result=$((10#$a - 10#$b))

    # Optional: clamp negative result to 0
    if (( result < 0 )); then
        result=0
    fi
    
    # Print zero-padded 4-digit result
    printf "%04d\n" "$result"
}

add_4digit() {
    local a="$1"
    local b="$2"

    # Convert to base-10 to avoid octal issues
    local result=$((10#$a + 10#$b))

    # Print zero-padded 4-digit result
    printf "%04d\n" "$result"
}


# Créer le répertoire de sortie s'il n'existe pas
mkdir -p "$output_dir"



# Fonction pour traiter chaque fichier
process_file() {
  local file="$1"
  local filename
  local base_dir

  filename=$(basename "$file")
  base_dir=$(dirname "$file")

  echo "Processing: $filename"

  # Format attendu :
  # LHD_FXX_0656_6862_PTS_LAMB93_IGN69.copc_0_18.laz
  if [[ "$filename" =~ ^(LHD_FXX_)([0-9]+)_([0-9]+)(_PTS_LAMB93_IGN69\.copc)_([0-9]+)_([0-9]+)\.laz$ ]]; then

    local prefix="${BASH_REMATCH[1]}"
    local tx="${BASH_REMATCH[2]}"
    local ty="${BASH_REMATCH[3]}"
    local suffix="${BASH_REMATCH[4]}"
    local xx="${BASH_REMATCH[5]}"
    local yy="${BASH_REMATCH[6]}"

    echo "  Tile: TX=$tx TY=$ty xx=$xx yy=$yy"

  else
    echo "ERROR: nom de fichier inattendu : $filename"
    return 1
  fi

  # Convertir explicitement en base 10
  tx=$((10#$tx))
  ty=$((10#$ty))
  xx=$((10#$xx))
  yy=$((10#$yy))

  neighbors=()

  # Chercher les tuiles dans un voisinage de 5x5
  for dx in -2 -1 0 1 2; do
    for dy in -2 -1 0 1 2; do

      nx=$((xx + dx))
      ny=$((yy + dy))
      ntx=$tx
      nty=$ty

      # Passage à la dalle précédente/suivante
      if (( nx < 0 )); then
        nx=$((nx + 50))
        ntx=$((tx - 1))
      elif (( nx > 49 )); then
        nx=$((nx - 50))
        ntx=$((tx + 1))
      fi

      if (( ny < 0 )); then
        ny=$((ny + 50))
        nty=$((ty - 1))
      elif (( ny > 49 )); then
        ny=$((ny - 50))
        nty=$((ty + 1))
      fi

      # Format 4 chiffres pour les coordonnées de dalle
      ntx_fmt=$(printf "%04d" "$ntx")
      nty_fmt=$(printf "%04d" "$nty")

      neighbor="LHD_FXX_${ntx_fmt}_${nty_fmt}_PTS_LAMB93_IGN69.copc_${nx}_${ny}.laz"

      neighbors+=("$neighbor")
    done
  done

  # Chercher les fichiers existants
  input_files=()

  for neighbor in "${neighbors[@]}"; do
    local match="$base_dir/$neighbor"

    if [[ -f "$match" ]]; then
      input_files+=("$match")
    fi
  done

  echo "  Found ${#input_files[@]} neighboring files"

  # Il faut au minimum le fichier courant
  if [[ ${#input_files[@]} -eq 0 ]]; then
    echo "ERROR: aucun fichier voisin trouvé pour $filename"
    return 1
  fi

  # Construire le nom de sortie
  local output_file="${output_dir}/${filename}"

  echo "  Output: $output_file"

  # Construire le tableau des readers PDAL
  readers_json=""

  for input_file in "${input_files[@]}"; do
    readers_json+="
    {
      \"type\": \"readers.las\",
      \"filename\": \"${input_file}\"
    },"
  done

  # Supprimer la dernière virgule
  readers_json="${readers_json%,}"

  pipeline=$(mktemp)

  cat > "$pipeline" <<EOF
{
  "pipeline": [
    ${readers_json},
    {
      "type": "writers.las",
      "filename": "${output_file}",
      "compression": "laszip"
    }
  ]
}
EOF

  # Afficher le pipeline en cas de problème
  echo "  Running PDAL..."

  if ! pdal pipeline "$pipeline"; then
    echo "ERROR: PDAL failed for $filename"
    echo "Pipeline:"
    cat "$pipeline"
    rm -f "$pipeline"
    return 1
  fi

  rm -f "$pipeline"

  # Statistiques
  stat_file="${output_file%.laz}.txt"
  pdal info --metadata "$output_file" > "$stat_file"

  echo "  Done: $filename"
}


export -f process_file
export output_dir

# Limiter le nombre de processus en arrière-plan
# Get the total number of CPU cores
total_cores=$(nproc)
max_jobs=8 #$((total_cores - 1))

count=0

for file in "${input_dir}"/*.laz; do
  process_file "$file" &
  count=$((count + 1))

  # Attendre si le nombre maximum de tâches est atteint
  if [[ $count -ge $max_jobs ]]; then
    wait -n
    count=$((count - 1))
  fi
done

# Attendre la fin de toutes les tâches en arrière-plan
wait


exit 0
