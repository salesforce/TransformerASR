# Large-scale ASR training

In this folder (`egs/combine/asr1/`), we provide a recipe for large-scale ASR training. Note that the core training and decoding components are the same as described in other parts of the packages (e.g., the offline case is described in the `egs/swbd/asr1/` folder, and the online case is described in the `egs/librispeech/asr1/` folder). Here we mainly focus on the data preparation part.

The data preparation is done in the `prepare.sh` script.

* The first step is offline feature computation. This step is mostly done with the `speech-datasets` package. We change to the dataset folders of `speech-datasets` and issue the `run.sh` script with the desired feature type. We created two special `feats_type` named `fbank8k` and `fbank16k`, which makes sure that audio clips are resampled to the specified frequency before extracting the filter bank features. In the example script we also specified `--remove_short_from_test true --max_frame_number 3000` which removes utterances with more than 3000 frames (as the frame length is 10ms, this corresponds to 30s and is sufficiently large to cover most utterances). Removing the extremely long utterances during training/validation keeps the GPU memory consumption under control. It is also possible to manipulate the audio clips, e.g., by adding speed perturbations or mixing with noise, as enabled by `speech-datasets`. By default, the offline features are dumped to the corresponding dataset folders under `speech-datasets`; this is fine as the dataloader provided by `speech-datasets` will be able to locate them. A global cmvn file extracted by combining the cmvn files for each dataset. 

* The second step is to gather the transcriptions of all training sets together, and extract the subword token set.

* Optionally, the last step is to train a RNNLM using the training text and perhaps additional text data. This step is time consuming, and an observation is that with large training corpus, the importance of the subword language model decreases.

Without using audio-level augmentation, we obtain a large training set of 5920 hours of audio data, with the following corpora:

| dataset                    | hours |
| :-----------------------   | ----:  |
| swbd | 285|
| fisher | 1905 | 
| librispeech | 960 |
| commonvoice | 875 |
| internal collection | 1895|
| **total**   | **5920** | 

## Results

We use the above combined dataset for training both offline models and online models. The token set consists of 4000 BPEs. The model architectures are as follows:

| model                    | architecture |
| :-----------------------   | :----  |
| offline | #enc layers: 24, #dec layers: 12, attention dim: 64 x 8,  FFN dim: 2048, input subsampling: 2 conv2d layers, subsampling rate: 4 |
| online  | #enc layers: 15, #dec layers:  9, attention dim: 64 x 12, FFN dim: 2048, input subsampling: frame stacking,  subsampling rate: 3 |
|         | enc left context: 1.8s, audio input block size: 0.45s |
| RNNLM   | 2 x 2048 uniLSTM |  

We evaluate on the `swbd/eval2000` set, and our recipes obtain the following WERs.

|              |SWBD (%) |CALLHM (%) |
| :--------    | :----:  | :----:    |
| offline model                   | 7.4     | 11.5      |
| online model, offline decode    | 8.2     | 12.4      |
| online model, CTC decode        | 11.8    | 15.8      |
| online model, online decode     | 10.4    | 13.6      |
