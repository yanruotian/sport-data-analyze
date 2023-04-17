#!/bin/bash

function Set-Variable () {
    local _varname="$1"
    local _value="$2"
    eval $_varname="'$_value'"
}

Set-Variable fake 1

which python

# run bs lr seed
function run() {
    Set-Variable BATCH_SIZE $1
    Set-Variable LR $2
    Set-Variable SEED $3

    Set-Variable DATASET "conHR"
    Set-Variable MODEL_TYPE "lstm"
    Set-Variable CRITERIA "macro-f1"

    if [ -z ${fake} ]; then
        Set-Variable INPUT_PATH "datasets/splited/${DATASET}"
        Set-Variable OUTPUT_DIR "outputs/${DATASET}/${MODEL_TYPE}/criteria=${CRITERIA}/bs=${BATCH_SIZE}/lr=${LR}/seed=${SEED}"
    else
        Set-Variable INPUT_PATH "datasets/splited/${DATASET}/fake"
        Set-Variable OUTPUT_DIR "outputs/${DATASET}-fake/${MODEL_TYPE}/criteria=${CRITERIA}/bs=${BATCH_SIZE}/lr=${LR}/seed=${SEED}"
    fi

    echo $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    Set-Variable OPTS ""
    OPTS+=" --criteria ${CRITERIA}" 
    OPTS+=" --model-type ${MODEL_TYPE}" 
    OPTS+=" --seed ${SEED}" 
    OPTS+=" --lr ${LR}" 
    OPTS+=" --batch-size ${BATCH_SIZE}" 
    OPTS+=" --epoch 50" 
    OPTS+=" --input-path ${INPUT_PATH}" 
    OPTS+=" --output-path ${OUTPUT_DIR}" 
    OPTS+=" --seq-len 10" 
    OPTS+=" --eval-step 100" 
    OPTS+=" --device cuda" 
    OPTS+=" --train-proportion 0.6" 
    OPTS+=" --early-stopping 10" 
    OPTS+=" --dataset ${DATASET}"

    Set-Variable CMD "python -m sport ${OPTS}"
    Set-Variable LOG_PATH "${OUTPUT_DIR}/run.log"
    echo "${CMD}"
    echo "log path = ${LOG_PATH}"
    ${CMD} | tee ${LOG_PATH}
}

function runLr() {
    Set-Variable LR $1
    for bs in 64 128; do
        for seed in 42 100 1768; do
            run ${bs} ${LR} ${seed}
        done
    done
}

export CUDA_VISIBLE_DEVICES=0
runLr "5e-3" &

export CUDA_VISIBLE_DEVICES=1
runLr "1e-3" &

export CUDA_VISIBLE_DEVICES=2
runLr "5e-4" &

export CUDA_VISIBLE_DEVICES=3
runLr "1e-4" &

export CUDA_VISIBLE_DEVICES=4
runLr "5e-2" &

export CUDA_VISIBLE_DEVICES=5
runLr "1e-2" &

wait
