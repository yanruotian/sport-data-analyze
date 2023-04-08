where.exe python.exe

Set-Variable LR "1e-5"
Set-Variable SEED 100
Set-Variable MODEL_TYPE "lstm"
Set-Variable CRITERIA "macro-f1"
Set-Variable BATCH_SIZE 3
Set-Variable INPUT_PATH "splited/综合数据"
Set-Variable OUTPUT_DIR "outputs/${MODEL_TYPE}/criteria=${CRITERIA}/lr=${LR}/seed=${SEED}"

Write-Output $OUTPUT_DIR

if (-not (Test-Path $OUTPUT_DIR)) {
    mkdir -p $OUTPUT_DIR
}

Set-Variable OPTS (
    " --criteria ${CRITERIA}" +
    " --model-type ${MODEL_TYPE}" +
    " --seed ${SEED}" +
    " --lr ${LR}" +
    " --batch-size ${BATCH_SIZE}" +
    " --epoch 50" +
    " --input-path ${INPUT_PATH}" +
    " --output-path ${OUTPUT_DIR}" +
    " --seq-len 5" +
    " --eval-step 100" +
    " --device cuda" +
    " --train-proportion 0.6" +
    " --early-stopping 10"
)

Set-Variable CMD "python.exe -m sport ${OPTS}"
Write-Output ${CMD}
powershell.exe -Command "${CMD} > ${OUTPUT_DIR}/run.log"
