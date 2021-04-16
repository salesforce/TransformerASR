#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2020 Salesforce Research (Weiran Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
gpuid=
ncpu=64        # number of cpus, depends on the machine type you get
debugmode=1
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
preprocess_config=conf/specaug_dsl.yaml
fisher_dir=""

# training configuration
train_config=conf/train_largestoc5.yaml
maxepoch=150

# input subsampling
input_layer=conv2d
input_context=0
input_skiprate=1

# rnnlm related
rnnlm_config=conf/rnnlm.yaml
rnnlm_resume= # specify a snapshot file to resume LM training
lmtag=lm     # tag for managing LMs
rnnlm_weight=0.2

# decoding parameter
decode_config=conf/decode.yaml
# transformer-based model performs averaging of last 10 snapshot
n_average=10
lang_model=rnnlm.model.best # set a language model to be used for decoding
beamsize=20

# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

# decode set
decode_set="eval2000 rt03"

# The feature directory created by speech-datasets.
dumpdir=/export/home/speech-datasets/swbd/asr1/dump/fbank_pitch

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=${gpuid}

train_set=swbd1_train
train_dev=swbd1_dev

# The training/dev/test fbank_pitch features are generated by the speech-datasets package,
# which alleviates us from generating the json files for inputs.
# This is done by going into the swbd/asr1/ folder of speech-datasets/, and running
# ./run.sh --feats_type fbank_pitch
# The dumpdir/ directory then contains ark features and text files.
# Also, change the path to cmvn.ark in your preprocess_config into the one prepared by speech-datasets.

mkdir -p data/lang_char/
dict=data/lang_char/train_nodup_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/train_nodup_${bpemode}${nbpe}
echo "dictionary: ${dict}"
for x in swbd1_train swbd1_dev eval2000 rt03; do
  # map acronym such as p._h._d. to p h d
  # Change of <noise> to [noise] is to accommodate the previously extracted bpe model which is trained with [noise].
  sed 's/\._/ /g; s/\.//g; s/them_1/them/g; s/<noise>/[noise]/g; s/<vocalized-noise>/[vocalized-noise]/g; s/<laughter>/[laughter]/g' \
  ${dumpdir}/${x}/text > ${dumpdir}/${x}/text_char
done

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "stage 2: Dictionary Preparation"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    echo "make a dictionary"
    cut -f 2- -d" " ${dumpdir}/${train_set}/text_char > data/lang_char/input.txt

    # Please make sure sentencepiece is installed
    spm_train --input=data/lang_char/input.txt \
            --model_prefix=${bpemodel} \
            --vocab_size=${nbpe} \
            --character_coverage=1.0 \
            --model_type=${bpemode} \
            --model_prefix=${bpemodel} \
            --input_sentence_size=100000000 \
            --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0 \
            --user_defined_symbols="[laughter],[noise],[vocalized-noise]"

    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${rnnlm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    mkdir -p data/local/lm_train ${lmdatadir}
    cut -f 2- -d" " ${dumpdir}/${train_set}/text_char | gzip -c > data/local/lm_train/${train_set}_text.gz
    if [ -n "${fisher_dir}" ]; then
        cut -f 2- -d" " ${fisher_dir}/text_char | gzip -c > data/local/lm_train/train_fisher_text.gz
        # combine swbd and fisher texts
        zcat data/local/lm_train/${train_set}_text.gz data/local/lm_train/train_fisher_text.gz |\
            spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    else
        zcat data/local/lm_train/${train_set}_text.gz |\
            spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    fi
    cut -f 2- -d" " ${dumpdir}/${train_dev}/text_char | \
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/valid.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${rnnlm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${rnnlm_resume} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expname=train_nodup_${backend}_$(basename ${train_config%.*})
    if [ -n "${preprocess_config}" ]; then
	    expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=train_nodup_${backend}_${tag}
fi
expname=${expname}_${bpemode}${nbpe}
expdir=exp/${expname}
mkdir -p ${expdir}

odim=$(wc -l ${dict} | awk '{print $1}')
odim=$(echo "${odim}+2" | bc)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    python -u -m torch.distributed.launch --nproc_per_node=${ngpu} \
        ../../../espnet/bin/asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --epochs ${maxepoch} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --verbose ${verbose} \
        --resume ${resume} \
        --transformer-input-layer ${input_layer} \
        --input-context ${input_context} \
        --input-skiprate ${input_skiprate} \
        --train-sets swbd/swbd1_train,swbd/swbd1_dev \
        --valid-sets swbd/rt03 \
        --idim 83 --odim ${odim} --precomputed-feats-type fbank_pitch \
        --text-filename text_char \
        --spmodel ${bpemodel}.model
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"

    nj=${ncpu}
    recog_model=model.last${n_average}.avg.best
    average_checkpoints.py --backend ${backend} \
         --snapshots ${expdir}/results/snapshot.ep.* \
         --out ${expdir}/results/${recog_model} \
         --num ${n_average}
    pids=() # initialize pids
    for rtask in ${decode_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}_subweight${rnnlm_weight}

        # split data
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu 0 \
            --backend ${backend} \
            --recog-sets swbd/${rtask} \
            --precomputed-feats-type fbank_pitch \
            --preprocess-conf ${preprocess_config} \
            --text-filename text_char \
            --spmodel ${bpemodel}.model \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model} \
            --lm-weight ${rnnlm_weight} \
            --input-context ${input_context} \
            --input-skiprate ${input_skiprate} \
            --num_replicas ${nj} \
            --jobid JOB \
            --result-label ${expdir}/${decode_dir}/data.JOB.json

    # Remove the extra dataset name from speechdataloader' uttids.
    sed -i "s|{${rtask}}||g" ${expdir}/${decode_dir}/data.*.json
    sed -i.org "s|{${rtask}}||g" ${dumpdir}/${rtask}/stm

	  # this is required for local/score_sclite.sh to get hyp.wrd.trn
	  score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
	  if [[ "${decode_dir}" =~ "eval2000" ]]; then
        local/score_sclite.sh ${dumpdir}/eval2000 ${expdir}/${decode_dir}
	  elif [[ "${decode_dir}" =~ "rt03" ]]; then
	      local/score_sclite.sh ${dumpdir}/rt03 ${expdir}/${decode_dir}
	  fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Network Training with wav inputs"

    dumpdir=/export/home/speech-datasets/swbd/asr1/dump/raw
    for x in swbd1_train swbd1_dev eval2000 rt03; do
      # map acronym such as p._h._d. to p h d
      # Change of <noise> to [noise] is to accommodate the previously extracted bpe model which is trained with [noise].
      sed 's/\._/ /g; s/\.//g; s/them_1/them/g; s/<noise>/[noise]/g; s/<vocalized-noise>/[vocalized-noise]/g; s/<laughter>/[laughter]/g' \
      ${dumpdir}/${x}/text > ${dumpdir}/${x}/text_char
    done

    python -u -m torch.distributed.launch --nproc_per_node=${ngpu} \
        ../../../espnet/bin/asr_train.py \
        --config ${train_config} \
        --epochs ${maxepoch} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --verbose ${verbose} \
        --resume ${resume} \
        --transformer-input-layer ${input_layer} \
        --input-context ${input_context} \
        --input-skiprate ${input_skiprate} \
        --train-sets swbd/swbd1_train,swbd/swbd1_dev \
        --valid-sets swbd/rt03 \
        --idim 80 --odim ${odim} \
        --precomputed-feats-type raw \
        --preprocess-conf conf/fbank_specaug_dsl.yaml \
        --text-filename text_char \
        --spmodel ${bpemodel}.model \
        --maxlen-in 480
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding with raw wav inputs"

    nj=${ncpu}
    recog_model=model.last${n_average}.avg.best
    average_checkpoints.py --backend ${backend} \
         --snapshots ${expdir}/results/snapshot.ep.* \
         --out ${expdir}/results/${recog_model} \
         --num ${n_average}
    pids=() # initialize pids
    for rtask in ${decode_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}_subweight${rnnlm_weight}

        # split data
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu 0 \
            --backend ${backend} \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model} \
            --lm-weight ${rnnlm_weight} \
            --input-context ${input_context} \
            --input-skiprate ${input_skiprate} \
            --num_replicas ${nj} \
            --jobid JOB \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --recog-sets swbd/${rtask} \
            --precomputed-feats-type raw \
            --preprocess-conf conf/fbank_specaug_dsl.yaml \
            --text-filename text_char \
            --spmodel ${bpemodel}.model

    # Remove the extra dataset name from speechdataloader' uttids.
    sed -i "s|{${rtask}}||g" ${expdir}/${decode_dir}/data.*.json
    sed -i.org "s|{${rtask}}||g" ${dumpdir}/${rtask}/stm

	  # this is required for local/score_sclite.sh to get hyp.wrd.trn
	  score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
	  if [[ "${decode_dir}" =~ "eval2000" ]]; then
        local/score_sclite.sh ${dumpdir}/eval2000 ${expdir}/${decode_dir}
	  elif [[ "${decode_dir}" =~ "rt03" ]]; then
	      local/score_sclite.sh ${dumpdir}/rt03 ${expdir}/${decode_dir}
	  fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

# For data loader
# ./run_char_bpe.sh --stage 3 --stop_stage 3 --ngpu 1 --gpuid 0 --tag speechdataloader  --verbose 1
# ./run_char_bpe.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --train_config conf/train_largestoc5.yaml --tag speechdataloader  --verbose 1
# ./run_char_bpe.sh --stage 5 --stop_stage 5 --tag speechdataloader --decode_set eval2000
# ./run_char_bpe.sh --stage 6 --stop_stage 6 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --train_config conf/train_largestoc5.yaml --tag waveform  --verbose 1
# ./run_char_bpe.sh --stage 7 --stop_stage 7 --tag waveform --decode_set eval2000