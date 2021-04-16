#!/bin/bash


#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


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
batch_size=12
accum_grad=8

# distributed multi-pods
nnodes=1
node_rank=0
master_addr=
master_port=1234

# input
idim=80

# acoustic modeling
train_config=conf/train_offline.yaml
ctc_type=warpctc

# language modeling
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lmexpdir=
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=1               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# bpemode
nbpe=4000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.
freq=16k

SPEECH_DATASETS=/export/home/adata
datasets8k="swbd fisher librispeech speechocean"
datasets16k="swbd fisher librispeech speechocean commonvoice"
valid_sets="swbd/rt03"

# decode set
decode_set="swbd/eval2000"

# input subsampling
input_layer=conv2d
input_context=0
input_skiprate=1

# streaming decode
ctc_weight=0.5
beamsize=30
rnnlm_weight=0.2
blank_offset=0.0
att_delay=7
dec_delay=0
length_bonus=1.0

. utils/parse_options.sh || exit 1;

export CUDA_VISIBLE_DEVICES=${gpuid}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

feats_type=fbank${freq}
preprocess_config=conf/specaug_dsl_${freq}.yaml
tag="${tag}${freq}"

if [ ${freq} = "8k" ]; then
    datasets=${datasets8k}
else
    datasets=${datasets16k}
fi
train_sets=""
for data in ${datasets}; do
    train_sets="${train_sets} ${SPEECH_DATASETS}/${data}/asr1/dump/fbank${freq}/*train*"
done
train_sets=$(echo ${train_sets} | sed "s:${SPEECH_DATASETS}/::g" | sed "s:asr1/dump/fbank${freq}/::g" | tr " " ",")
echo "training sets: ${train_sets}"
echo "validation sets: ${valid_sets}"

train_set=train_all
mkdir -p data/lang_char/
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"

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

    # --nnodes=${nnodes} --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} \
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
        --transformer-input-layer ${input_layer} \
        --input-context ${input_context} \
        --input-skiprate ${input_skiprate} \
        --train-sets ${train_sets} \
        --valid-sets ${valid_sets} \
        --precomputed-feats-type ${feats_type} \
        --idim ${idim} --odim ${odim} \
        --preprocess-conf ${preprocess_config} \
        --text-filename text \
        --spmodel ${bpemodel}.model \
        --batch-size ${batch_size} \
        --accum-grad ${accum_grad}
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
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_lmweight${rnnlm_weight}_ctcweight${ctc_weight}_beamsize${beamsize}

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
            --lm-weight ${rnnlm_weight} \
            --ctc-weight ${ctc_weight} \
            --beam-size ${beamsize} \
            --num_replicas ${nj} \
            --jobid JOB \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --recog-sets ${rtask} \
            --precomputed-feats-type ${feats_type} \
            --preprocess-conf ${preprocess_config} \
            --text-filename text \
            --spmodel ${bpemodel}.model \
            --input-context ${input_context} \
            --input-skiprate ${input_skiprate}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

        if [[ "${decode_dir}" =~ "eval2000" ]]; then
            # Remove the extra dataset name from speechdataloader' uttids.
            sed -i "s|{eval2000}||g; s/\._/ /g; s/\.//g; s/<noise>/[noise]/g" ${expdir}/${decode_dir}/data.*.json
            sed -i "s|{eval2000}||g" ${SPEECH_DATASETS}/swbd/asr1/dump/fbank${freq}/eval2000/stm

            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
            local/score_sclite.sh ${SPEECH_DATASETS}/swbd/asr1/dump/fbank${freq}/eval2000 ${expdir}/${decode_dir}
        fi

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
            --recog-sets ${rtask} \
            --precomputed-feats-type ${feats_type} \
            --preprocess-conf ${preprocess_config} \
            --text-filename text \
            --spmodel ${bpemodel}.model \
            --input-context ${input_context} \
            --input-skiprate ${input_skiprate}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

        if [[ "${decode_dir}" =~ "eval2000" ]]; then
            # Remove the extra dataset name from speechdataloader' uttids.
            sed -i "s|{eval2000}||g; s/\._/ /g; s/\.//g; s/<noise>/[noise]/g" ${expdir}/${decode_dir}/data.*.json
            sed -i "s|{eval2000}||g" ${SPEECH_DATASETS}/swbd/asr1/dump/fbank${freq}/eval2000/stm

            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
            local/score_sclite.sh ${SPEECH_DATASETS}/swbd/asr1/dump/fbank${freq}/eval2000 ${expdir}/${decode_dir}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

# ./run.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --tag offline --freq 8k  --verbose 1
# ./run.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --tag offline --freq 16k --verbose 1
#
# ./run.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --train_config conf/train_streaming_block45.yaml --input_layer linear --input_context 1 --input_skiprate 3 --tag streaming --freq 8k  --verbose 1
# ./run.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --train_config conf/train_streaming_block45.yaml --input_layer linear --input_context 1 --input_skiprate 3 --tag streaming --freq 16k --verbose 1
#
# ./run.sh --stage 5 --stop_stage 5 --train_config conf/train_streaming_block45.yaml --input_layer linear --input_context 1 --input_skiprate 3 --tag streaming --freq 8k --lmexpdir exp/train_rnnlm_pytorch_lm_bpe4000
# ./run.sh --stage 6 --stop_stage 6 --train_config conf/train_streaming_block45.yaml --input_layer linear --input_context 1 --input_skiprate 3 --tag streaming --freq 8k --lmexpdir exp/train_rnnlm_pytorch_lm_bpe4000