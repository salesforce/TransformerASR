#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


# Copyright 2020 Salesforce Research (Weiran Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script contains all the data preparation commands.

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
stop_stage=100
ngpu=8         # number of gpus ("0" uses cpu, otherwise use gpu)
gpuid=
ncpu=64

SPEECH_DATASETS=/export/home/adata

# bpemode
nbpe=4000
bpemode=bpe
train_set=train_all
lm_config=conf/lm.yaml
lm_resume=
# Add options to remove very long utterances.
asr_options="--remove_short_from_test true --max_frame_number 3000"

. utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    current_dir=`pwd`

    # swbd: 285 hours
    cd ${SPEECH_DATASETS}/swbd/asr1
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank16k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank16k ${asr_options}
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank8k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank8k ${asr_options}

    # fisher: 1905 hours
    cd ${SPEECH_DATASETS}/fisher/asr1
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank16k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank16k ${asr_options}
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank8k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank8k ${asr_options}

    # librispeech: 960 hours
    cd ${SPEECH_DATASETS}/librispeech/asr1
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank16k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank16k ${asr_options}
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank8k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank8k ${asr_options}

    # commonvoice: 875 hours
    cd ${SPEECH_DATASETS}/commonvoice/asr1
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank16k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank16k ${asr_options}
    # WEIRAN: THIS DOES NOT WORK.
    # cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank8k.yaml conf/
    # ./run.sh --nj ${ncpu} --feats_type fbank8k

    # speechocean: 1895 hours
    cd ${SPEECH_DATASETS}/speechocean/asr1
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank16k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank16k ${asr_options}
    cp ${SPEECH_DATASETS}/TEMPLATE/asr1/conf/fbank8k.yaml conf/
    ./run.sh --nj ${ncpu} --feats_type fbank8k ${asr_options}

    cd ${current_dir}

    # Combine cmvn stats.
    for freq in 8k 16k; do
        cmvns=""
        # left out commonvoice
        for data in swbd fisher librispeech speechocean; do
            cmvns="${cmvns} ${SPEECH_DATASETS}/${data}/asr1/dump/fbank${freq}/global_cmvn.ark"
        done
        python3 -m speech_datasets.bin.combine_cmvn_stats --cmvn_type global --output_file global_cmvn_${freq}.ark \
            ${cmvns}
    done

    # Prepare text.
    mkdir -p data/lang_char
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        alltext=""
        for data in swbd fisher librispeech speechocean commonvoice; do
            alltext="${alltext} ${SPEECH_DATASETS}/${data}/asr1/dump/fbank16k/*train*/text"
        done
        echo ${alltext}
        cut -f 2- -d" " ${alltext} > data/lang_char/input.txt
        # Check words with special characters
        tr " " "\n" < data/lang_char/input.txt | sort | uniq > data/lang_char/words.txt
    fi
fi


dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "Dictionary Preparation"
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    spm_train --input=data/lang_char/input.txt \
              --vocab_size=${nbpe} \
              --model_type=${bpemode} \
              --model_prefix=${bpemodel} \
              --input_sentence_size=100000000 \
              --character_coverage=1.0 \
              --bos_id=-1 \
              --eos_id=-1 \
              --unk_id=0 \
              --user_defined_symbols="<noise>"
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
fi


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
    for data in swbd fisher librispeech speechocean commonvoice; do
        cut -f 2- -d" " ${SPEECH_DATASETS}/${data}/asr1/dump/fbank16k/*train*/text
    done | gzip -c > data/local/lm_train/${train_set}_text.gz
    # use external data
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    # use voice corpus 2
    lbzip2 -dc /export/share/ykang/LMs/voice_corpus2/corpus.txt.bz2 | gzip -c > data/local/lm_train/voice_corpus2.gz
    # combine external text and transcriptions and shuffle them with seed 777
    # note that we use all lower case letters in speech-datasets, so the additional text data needs to be as well
    zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/voice_corpus2.gz data/local/lm_train/${train_set}_text.gz |\
        tr '[:upper:]' '[:lower:]' |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " ${SPEECH_DATASETS}/swbd/asr1/dump/fbank16k/rt03/text | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/valid.txt
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
