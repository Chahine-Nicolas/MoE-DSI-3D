#!/bin/bash

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR=$(realpath "$(dirname "$CUR_DIR")")
source ${MAIN_DIR}/common.sh
echo "======================"
echo $CUR_DIR
echo $MAIN_DIR
echo "-----"
#exit 0

usage() {
  echo "Usage: $0 --list_files <txt_file> --project_path <project_path>"
  exit 1
}

POW=3
CHUNK_SIZE=20
BBOX_SIZE=100
DO_CROP=false
DO_USE_DOCKER=true

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --list_files) LIST_FILES="$2"; shift ;;
    --bbox_size) BBOX_SIZE="$2"; shift ;;
    --pow) POW="$2"; shift ;;
    --center) CENTER_FILE="$2"; shift ;;      
    --do_crop) DO_CROP=true ;;
    --project_path) PROJECT_PATH="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

[ -z "$LIST_FILES" ] || [ -z "$PROJECT_PATH" ] && { echo "Error." ; usage ; }

# Afficher un résumé des paramètres
echo "Résumé des paramètres:"
echo "  - Fichier list_files  : $LIST_FILES"
echo "  - Fichier center      : ${CENTER_FILE:-Non spécifié}"
echo "  - Taille BBOX         : $BBOX_SIZE"
echo "  - Do_CROP         : $BBOX_SIZE"
echo "  - Chemin projet       : $PROJECT_PATH"


execute_command() {
  if [ "$DO_USE_DOCKER" = true ]; then
    log "Start ${CMD} ..."
    MSYS_NO_PATHCONV=1 docker run --rm \
       -v "${PROJECT_PATH}:${PROJECT_PATH}" \
       -v "${CUR_DIR}/scripts:/data/scripts" \
       pdal/pdal \
       /bin/bash -c "${CMD}"
    # docker run --rm -it -v ${PROJECT_PATH}:${PROJECT_PATH} \
    #        -v ${CUR_DIR}/scripts:/data/scripts \
    #        pdal/pdal \
    #        /bin/bash 
  else
    log "Start ${CMD} locally..."
    eval "${CMD}"
  fi
}



RAW_LAZ_DIR=${PROJECT_PATH}/raw_inputs

log "download ..." "$PROCESS"
mkdir -p ${RAW_LAZ_DIR}/inputs
while IFS= read -r line; do
  FILENAME=$(basename "${line}")
  FILEPATH=${RAW_LAZ_DIR}/${FILENAME}
  if [ ! -e "${FILEPATH}" ]; then
    curl -L -o "${FILEPATH}" "${line}"
  fi
done < "${LIST_FILES}"


log "compute stats ..." "$PROCESS"
INPUT_DIR=${RAW_LAZ_DIR}
STATS_DIR=${PROJECT_PATH}/stats
if [ -d "${STATS_DIR}" ]; then
  log "${STATS_DIR} exists, skip crop!" "$SKIP"
else
  mkdir -p ${STATS_DIR}

  #read -r LAT LONG < ${CENTER_FILE}
  log "Stats /data/scripts/tile_lidar_files.sh"
  CMD="/data/scripts/compute_stats.sh --input_dir=${INPUT_DIR} --output_dir=${STATS_DIR}"
  execute_command
fi
## Compute min 
read MIN_X MIN_Y <<< $(find ${STATS_DIR} -name "*.txt" -exec awk '{print $1, $2}' {} + | \
			 awk 'BEGIN {min1=9999999; min2=9999999} {if ($1 < min1) min1=$1} {if ($2 < min2) min2=$2} END {print min1, min2}')
log "MIN_X: $MIN_X, MIN_Y: $MIN_Y"
log "stats finished!" "$FINISH"


if [[ "$DO_CROP" == "true" ]]; then
  log "crop ..." "$PROCESS"
  CROPED_DIR=${PROJECT_PATH}/croped
  INPUT_DIR=${RAW_LAZ_DIR}
  if [ -d "${CROPED_DIR}" ]; then
    log "${CROPED_DIR} exists, skip crop!" "$SKIP"
  else
    mkdir -p ${CROPED_DIR}
    if [ -n "${BBOX_SIZE}" ]; then
      BBOX_SIZE_CMD="--bbox_size=${BBOX_SIZE}"
    else
      BBOX_SIZE_CMD=""
    fi

    if [ -n "${subsample_ratio}" ]; then
      SUBSAMPLE_RATIO_CMD="--subsample_ratio=${subsample_ratio}"
    else
      SUBSAMPLE_RATIO_CMD=""
    fi

    read -r LAT LONG < ${CENTER_FILE}
    CMD="/data/scripts/crop_lidar_files.sh --input_dir=${RAW_LAZ_DIR} --output_file=${CROPED_DIR}/cropped_merged.laz --LAT=${LAT} --LONG=${LONG} ${BBOX_SIZE_CMD} ${SUBSAMPLE_RATIO_CMD}"
    execute_command
    
  fi
  log "cropping finished!" "$FINISH"
fi


log "tiling ..." "$PROCESS"
if [[ "$DO_CROP" == "false" ]]; then
  INPUT_DIR=${RAW_LAZ_DIR}
else
  INPUT_DIR=${CROPED_DIR}
fi

PROCESSED_DIR=${PROJECT_PATH}/tiled
if [ -d "${PROCESSED_DIR}" ]; then
  log "${PROCESSED_DIR} exists, skip crop!" "$SKIP"
else
  mkdir -p ${PROCESSED_DIR}
  CMD="/data/scripts/tile_and_split.sh --input_dir=${INPUT_DIR} --output_dir=${PROCESSED_DIR} --min_x=${MIN_X}  --min_y=${MIN_Y} --chunk_size=${CHUNK_SIZE}"

  execute_command
fi
log "tiling finished!" "$FINISH"

MERGED_DIR=${PROJECT_PATH}/merged
if [ -d "${MERGED_DIR}" ]; then
  log "${MERGED_DIR} exists, skip crop!" "$SKIP"
else
  mkdir -p ${MERGED_DIR}
  CMD="/data/scripts/merge_lidar_file.sh --input_dir=${PROCESSED_DIR} --output_dir=${MERGED_DIR}"
  execute_command
fi
log "merging finished!" "$FINISH"



# log "convert to binary ..." "$PROCESS"
# NORMAL_DIR=${PROJECT_PATH}/merged_processed
# #rm -rf ${NORMAL_DIR}
# if [ -d "${NORMAL_DIR}" ]; then
#   log "${NORMAL_DIR} exists, skip crop!" "$SKIP"
# else
#   mkdir -p ${NORMAL_DIR}
#   echo "$NORMAL_DIR"
#   ${CUR_DIR}/scripts/compute_normals.sh --project_path=${PROJECT_PATH} --cgal_dir=${CUR_DIR}/cgal/ --output_dir=${NORMAL_DIR}
# fi
# log "convert done!" "$FINISH"

log "convert to bin  ..." "$PROCESS"
BIN_DIR=${PROJECT_PATH}/bin
if [ -d "${BIN_DIR}" ]; then
  log "${BIN_DIR} exists, skip crop!" "$SKIP"
else
  mkdir -p ${BIN_DIR}
  echo "$BIN_DIR"
  ${MAIN_DIR}/build/lidarhd_env/Scripts/python.exe ${CUR_DIR}/scripts/convert_bin.py --input_dir ${MERGED_DIR} --output_dir ${BIN_DIR}
fi
log "convert done!" "$FINISH"



## Stats
log "Stats /data/scripts/tile_lidar_files.sh"
CMD="/data/scripts/compute_stats_advanced.sh --input_dir=${MERGED_DIR} --output_dir=${MERGED_DIR}"
#execute_command

#${MAIN_DIR}/build/lidarhd_env/Scripts/python.exe   ${CUR_DIR}/scripts/plot_stats.py --input_dir ${MERGED_DIR}
#${MAIN_DIR}/build/lidarhd_env/Scripts/python.exe   ${CUR_DIR}/scripts/plot_dict.py --dict_path ${MERGED_DIR}/stats.json


