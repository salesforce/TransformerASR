#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2020 Salesforce Research (Weiran Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
gpuid=
ncpu=64
debugmode=1
verbose=0      # verbose option
resume=        # Resume the training from snapshot

preprocess_config=conf/specaug_dsl.yaml
train_config=conf/train_streaming_block40.yaml
ctc_type=warpctc

lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=10                 # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=6               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# bpemode
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

# decode set
decode_set="test-clean"

# input subsampling
input_layer=conv2d
input_context=0
input_skiprate=1

# streaming decode
ctc_weight=0.4
beamsize=50
rnnlm_weight=0.6
blank_offset=0.0
att_delay=7
dec_delay=0
length_bonus=1.0

# input
feats_type=fbank_pitch
idim=83

# The feature directory created by speech-datasets.
dumpdir=/export/home/speech-datasets/librispeech/asr1/dump/fbank_pitch

. utils/parse_options.sh || exit 1;

export CUDA_VISIBLE_DEVICES=${gpuid}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev-clean

# The training/dev/test fbank_pitch features are generated by the speech-datasets package,
# which alleviates us from generating the json files for inputs.
# This is done by going into the librispeech/asr1/ folder of speech-datasets/, and running
# ./run.sh --feats_type fbank_pitch
# The dumpdir/ directory then contains ark features and text files.
# Also, change the path to cmvn.ark in your preprocess_config into the one prepared by speech-datasets.

mkdir -p data/lang_char/
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " ${dumpdir}/train-clean-100/text ${dumpdir}/train-clean-360/text ${dumpdir}/train-other-500/text \
        > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt \
              --vocab_size=${nbpe} \
              --model_type=${bpemode} \
              --model_prefix=${bpemodel} \
              --input_sentence_size=100000000

    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    mkdir -p ${lmdatadir}
    # use external data
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    cut -f 2- -d" " ${dumpdir}/train-clean-100/text ${dumpdir}/train-clean-360/text ${dumpdir}/train-other-500/text |\
        gzip -c > data/local/lm_train/${train_set}_text.gz
    # combine external text and transcriptions and shuffle them with seed 777
    # note that we use all lower case letters in speech-datasets, so the additional text data needs to be as well
    zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
        tr '[:upper:]' '[:lower:]' |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " ${dumpdir}/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
        > ${lmdatadir}/valid.txt
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

odim=$(wc -l ${dict} | awk '{print $1}')
odim=$(echo "${odim}+2" | bc)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    python -u -m torch.distributed.launch --nproc_per_node=${ngpu} \
        ../../../espnet/bin/asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --verbose ${verbose} \
        --resume ${resume} \
        --ctc-type ${ctc_type} \
        --batch-count seq \
        --transformer-input-layer ${input_layer} \
        --input-context ${input_context} \
        --input-skiprate ${input_skiprate} \
        --train-sets librispeech/train-clean-100,librispeech/train-clean-360,librispeech/train-other-500 \
        --valid-sets librispeech/dev-clean,librispeech/dev-other \
        --precomputed-feats-type ${feats_type} \
        --idim ${idim} --odim ${odim} \
        --preprocess-conf ${preprocess_config} \
        --text-filename text \
        --spmodel ${bpemodel}.model
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Offline Decoding (attention proposes and ctc rescores)"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${lm_n_average}.avg.best
            opt="--log ${lmexpdir}/log"
        else
            lang_model=rnnlm.last${lm_n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=${ncpu}

    pids=() # initialize pids
    for rtask in ${decode_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}_ctcweight${ctc_weight}_beamsize${beamsize}

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model} \
            --ctc-weight ${ctc_weight} \
            --beam-size ${beamsize} \
            --num_replicas ${nj} \
            --jobid JOB \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --recog-sets librispeech/${rtask} \
            --precomputed-feats-type ${feats_type} \
            --preprocess-conf ${preprocess_config} \
            --text-filename text \
            --spmodel ${bpemodel}.model

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Simulated Online Decoding (ctc proposes and attention rescores)"

    nj=${ncpu}
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${lm_n_average}.avg.best
            opt="--log ${lmexpdir}/log"
        else
            lang_model=rnnlm.last${lm_n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    pids=() # initialize pids
    for rtask in ${decode_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}_subweight${rnnlm_weight}_ctcweight${ctc_weight}_blankoffset${blank_offset}_attdelay${att_delay}_bonus${length_bonus}_decdelay${dec_delay}_beamsize${beamsize}

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_streaming.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model} \
            --input-context ${input_context} \
            --input-skiprate ${input_skiprate} \
            --lm-weight ${rnnlm_weight} \
            --ctc-weight ${ctc_weight} \
            --streaming-ctc-blank-offset ${blank_offset} \
            --streaming-att-delay ${att_delay} \
            --streaming-dec-delay ${dec_delay} \
            --beam-size ${beamsize} \
            --penalty ${length_bonus} \
            --num_replicas ${nj} \
            --jobid JOB \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --recog-sets librispeech/${rtask} \
            --precomputed-feats-type ${feats_type} \
            --preprocess-conf ${preprocess_config} \
            --text-filename text \
            --spmodel ${bpemodel}.model

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

# ./run_streaming.sh --stage 3 --stop_stage 3 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7
# ./run_streaming.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --tag streaming --verbose 1
# ./run_streaming.sh --stage 5 --stop_stage 5 --tag streaming --decode_set test-clean
#
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.4
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.5
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.6
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.7
#
# Pure CTC.
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 1.0
#
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.5
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.5
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.5
# ./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.5
