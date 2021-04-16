# Transformer-ASR: end-to-end speech recognition with transformers

Transformer-ASR is an end-to-end automatic speech recognition toolkit. 
It is mostly built on top of ESPnet (version 1) developed by the authors of [1].
The loss for training the acoustic model is the multi-task CTC/Attention loss developed in [3]. 
The neural architecture we use is transformer-based encoder-decoder as presented in [4], while we have added the "stochastic layer" regularization technique from [5].
Another significant addition to ESPnet is the implementation of phone-based BPE systems, including the decoding algorithms developed in [6]. 
Same as ESPnet, we use [pytorch](http://pytorch.org/) as a main deep learning engine, and implement the Distributed DataParallel scheme for efficient training.
We provide complete recipes for the Switchboard corpus.

* [Installation](#installation)
* [Execution of example scripts](#execution-of-example-scripts)
* [Results](#results)
* [Expanding to other datasets](#expand)
* [References](#references)

## Installation

Go to the `tools/` directory and `make`. This would create a virtual environment, and install ESPnet and [warp-ctc](https://github.com/baidu-research/warp-ctc).  The 
pytorch version is set to 1.4 to allow finer control of DistributedDataParallel for multi-GPU training, and the 
warp-ctc's [pytorch binding](https://github.com/t-vi/warp-ctc) is updated to be compatible with the pytorch 1.4 version. 

To be able to run the scoring scripts (e.g., for `swbd` recipe), run `make sctk` to install sctk. 

### External dependency

We use the feature extraction and the pytorch dataloader provided by the [speech-datasets](https://github.com/salesforce/speech-datasets)
package for training. Please install the speech-datasets after installing transformer-ASR. Use the CONDA path 
and virtual environment name of transformer-ASR when installing speech-datasets, as detailed in the "Environment Setup" 
section of its [installation instructions](https://github.com/salesforce/speech-datasets/blob/master/README.md#environment-setup).

```bash
# run this in SpeechDataset repository folder
make clean all CONDA=<transformer-asr_root>/tools/venv/bin/conda VENV_NAME=base TORCH_VERSION=1.4.0

```
When prompted, continue with reusing the conda environment.

## Execution of example scripts

We provide complete recipes for the Switchboard (300 hours) corpus. Go to the `egs/swbd/asr1/` directory. The `run_char_bpe.sh` 
and `run_phone_bpe.sh` scripts contains the training steps (for both acoustic model and RNN language model) + decoding steps 
for the character BPE and phone BPE systems respectively. See the beginning of the scripts for data preparation instructions, 
and point the ``dumpdir`` variable to your prepared data folder using [speech-datasets](https://github.com/salesforce/speech-datasets).

Important modeling and decoding parameters can be configured; see the scripts for the complete list of options. For example, 
we use 2000 character BPEs and 500 phone BPEs by default as tuned in [6], but you can configure them by providing the `--nbpe` option.
Training requires at least one GPU, and you can configure the number of GPUs (`--ngpu`) and the IDs of available GPUs (`--gpuid`).

Make sure that this repo is added in the `PYTHONPATH` environment variable. For the character BPE recipe:

```bash
# char BPE RNNLM preparation
./run_char_bpe.sh --stage 3 --stop_stage 3 --ngpu 1 --gpuid 0 --tag speechdataloader  --verbose 1

# acoustic model training, takes 43 hours with 8 Tesla V100 GPUs (the GCP pod is equipped with 400G RAM)
./run_char_bpe.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --train_config conf/train_largestoc5.yaml --tag speechdataloader  --verbose 1

# decoding for eval2000, takes 1.5 hours with 64 CPUs
./run_char_bpe.sh --stage 5 --stop_stage 5 --ncpu 64 --tag speechdataloader --decode_set eval2000

# decoding for rt03, takes 2.5 hours with 64 CPUs
./run_char_bpe.sh --stage 5 --stop_stage 5 --ncpu 64 --tag speechdataloader --decode_set rt03
```

For the phone BPE recipe (for improved accuracy, specify in `fisher_dir` the folder that contains text data from fisher, 
and set `use_fisher_wordlm` to `true` to reproduce the results in table):

```bash
# phone BPE RNNLM preparation
./run_phone_bpe.sh --stage 3 --stop_stage 3 --ngpu 1 --gpuid 0 --tag speechdataloader  --verbose 1

# word RNNLM preparation
./run_phone_bpe.sh --stage 4 --stop_stage 4 --ngpu 1 --gpuid 0 --tag speechdataloader  --verbose 1

# acoustic model training, takes 43 hours with 8 Tesla V100 GPUs
./run_phone_bpe.sh --stage 5 --stop_stage 5 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --train_config conf/train_largestoc5.yaml --tag speechdataloader  --verbose 1

# decoding with phone BPE system, takes 3 hours with 95 CPUs
./run_phone_bpe.sh --stage 6 --stop_stage 6 --ncpu 64 --tag speechdataloader --decode_set eval2000

# joint decoding with both phone BPE and char BPE systems, takes 15 hours with 95 CPUs
./run_phone_bpe.sh --stage 7 --stop_stage 7 --ncpu 64 --tag speechdataloader --decode_set eval2000
```

Note that the decoding times are somewhat long, since the decoder that mostly inherited the structure of ESPNet's decoder, 
tried to be as simple as possible. With careful batching at the hypothesis level and utterance level, the decoding time 
can be significantly improved, see the [fast-beam-search](https://github.com/MetaMind/fast-beam-search) package for such 
implementations.  

## Results
With the efficient dataloader provided by the speech-datasets package, distributed training of the acoustic models takes
43 hours (for 150 training epochs) with 8 Tesla V100 GPUs, for each BPE system. The word error rates (WERs) on the eval2000
and rt03 sets are given in the following table (*offline fbank_pitch features* section) for the BPE systems.

We also provided example recipe with online feature computation functionality provided by the [speech-datasets](https://github.com/salesforce/speech-datasets) package, 
in stages 6 and 7 of `run_char_bpe.sh`. The training time using online feature computation is longer than using offline 
computed and stored features: with 8 Tesla V100 GPUs, training takes about 50 hours (for 150 training epochs). The potential 
advantage of online feature computation is that it takes little storage (without saving the extracted feature) and allows 
for perturbation at the audio level. The decoding results of the online feature system is provided in the following table 
(*online fbank features* section).
 
|              |SWBD (%) |CALLHM (%) |  RT03 (%) |
| :--------    | :----:  | :----:    |   :----:  |
| *offline fbank_pitch features* ||
| char BPE     | 7.0     | 14.4      |    12.8   |
| phone BPE    | 6.7     | 14.3      |    12.5   |
| joint decode | 6.3     | 13.3      |    11.4   |
| *online fbank features (no pitch)* ||
| char BPE     | 7.1     | 15.1      |    22.8   |

## Expanding to other datasets

We provide pointers for how to expand the swbd recipe to other speech datasets than `swbd`. In general, this can be done by 
copying the `run_char_bpe` and `run_phone_bpe.sh` scripts to a new folder, and modify the locations to prepared data (the variable 
``dumpdir``) in the scripts. For the phone recipe, the user needs to additionally provide the phone set (we generated it in 
``data/local/dict_phone/phones.txt`` for the `swbd` recipe) and the pronunciation dictionary (lexicon, we provided it in 
``data/local/dict_nosp/lexicon.txt`` for the `swbd` recipe). The lexicon and phone set are usually generated by kaldi-style 
data preparation steps that ESPNet and speech-datasets have followed. Alternatively, one can generate the lexicon with 
grapheme-to-phoneme methods (we provided one that extends the original 30K-word lexicon for `swbd` in ``data/extended_lexicon`` 
with one such method).

## Other features
We provide [streaming](egs/librispeech/asr1/streaming.md) and [non-autogreressive decoding](egs/librispeech/asr1/nonar.md) features. 
For more details please refer to detailed instructions in the 
examples for librispeech under ``egs/librispeech/asr1``.

## References

[1] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-End Speech Processing Toolkit". *Proc. Interspeech'18*, pp. 2207-2211 (2018).

[2] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning". *Proc. ICASSP'17*, pp. 4835--4839 (2017).

[3] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition". *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017.

[4] Shigeki Karita, Nanxin Chen, Tomoki Hayashi, Takaaki Hori, Hirofumi Inaguma, Ziyan Jiang, Masao Someki, Nelson Enrique Yalta Soplin, Ryuichi Yamamoto, Xiaofei Wang, Shinji Watanabe, Takenori Yoshimura, Wangyou Zhang, "A Comparative Study on Transformer vs RNN in Speech Applications". *IEEE Automatic Speech Recognition and Understanding Workshop*, 2019. 

[5] Ngoc-Quan Pham, Thai-Son Nguyen, Jan Niehues, Markus Muller, Sebastian Stuker, Alexander Waibel, "Very Deep Self-Attention Networks for End-to-End Speech Recognition". *Interspeech*, 2019.

[6] Weiran Wang, Guangsen Wang, Aadyot Bhatnagar, Yingbo Zhou, Caiming Xiong, Richard Socher, "An investigation of phone-based subword units for end-to-end speech recognition". *Interspeech*, 2020.
