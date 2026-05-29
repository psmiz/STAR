PILE_RAW=${PILE_RAW:-/path/to/pile_jsonl}
PILE_OUT=${PILE_OUT:-/path/to/pile_gpt_test}

mkdir -p "$PILE_OUT"
for i in $(seq -w 0 29); do
    python tools/preprocess_data.py \
        --input ${PILE_RAW}/${i}.jsonl \
        --output-prefix ${PILE_OUT}/${i} \
        --vocab-file gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file gpt2-merges.txt \
        --append-eod \
        --workers 32
done
