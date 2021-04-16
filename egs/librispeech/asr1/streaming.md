# Streaming ASR

## Training

Our package provides the capability of streaming ASR with the transformer-based, multi-task Attention/CTC modeling framework [1]. The training recipe for streaming ASR is mostly the same as that of the offline case. In fact the model definition is the same for both cases, as implemented by `e2e_asr_transformer_stoc.py` under `espnet/nets/pytorch_backend/`, and the interface for training is identical for both. The only difference for training is that we introduce causal masks when forwarding the encoder and the decoder, so that they only attend to past information. One can customize the history length by passing the following parameters to `e2e_asr_transformer_stoc.py`: 

```
'--streaming_encoder': Whether to use streaming transformer encoder

'--streaming_left_context': How many previous frames to attend to by encoder

'--streaming_right_context': How many future frames to attend to by encoder

'--streaming_block_size': How many frames are given to encoder at once

'--streaming_dec_context': How many previous tokens to attend to by decoder
``` 

Note that `--streaming_block_size` handles the scenario where the inputs frames are fed in chunks, which allows the encoder to use full-history attention *within* the chunk.

## Decoding

The decoding recipe for the online system is different from that of the offline system, and in some sense they are the opposite. In the offline case, we use the Attention module to propose the next token conditioned on the current partial hypothesis, and use the CTC module to score the new hypothesis; the calculations of the scores depends on the outputs of the encoder and decoder for the entire utterance (which is of course, impossible for the online case). The combined score from both modules are used for pruning in `label-synchronous` beam search. 
 
In the online case, one can instead use CTC to propose hypotheses and use Attention to rescore them on the fly; this is the approach taken by [2] which we have implemented here. The online decoding interface is `asr_recog_streaming.py` under `espnet/bin/`, while the core algorithm is implemented in `decode_streaming.py` under `espnet/asr/pytorch_backend/`. In more detail, we use the `frame-synchronous` beam search algorithm of CTC to maintain a set of hypotheses at each *frame*; each hypothesis contains a label sequence which removes repetition and blanks from the corresponding frame alignment. In the CTC beam search algorithm, we also maintain the time point (or frame index) when a token is first added; despite of the CTC alignment being not very accurate, this time point gives us a good idea of the starting point of this token. After a specified number of frames since the token is first added, as specified in the `--streaming-att-delay` parameter, we consider that token to have ended, at which point we invoke the decoder module of Attention to provide the score of this token conditioned on previous partial hypothesis (and the decoder attends to encoder outputs we have thus far). The combined score from both CTC and attention are used for pruning in the `frame-synchronous` beam search. 

The actual algorithm is somewhat more involved than described above. For example, the tokens can have variable time spans, and we may not wait for `streaming-att-delay` to have Attention score a new token. Also, sometimes the initial time point of the token reported by CTC is not accurate and can be updated as we accumulate more evidence. These aspect are handled with a few useful heuristics by [2] and we have implemented them as well. Care has been take to convert the pseudo code of [2] into an actual streaming algorithm, so that at any time point, we do not use information from the future unseen frames.

## Results

We provide the training and decoding recipes for the `Librispeech` corpus. Here are some example commands: 
 
```bash
# Acoustic model training 
./run_streaming.sh --stage 4 --stop_stage 4 --ngpu 8 --gpuid 0,1,2,3,4,5,6,7 --tag streaming --verbose 1

# Offline decoding of online model
./run_streaming.sh --stage 5 --stop_stage 5 --tag streaming --decode_set test-clean

# Online decoding of online model
./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 0.4

# Online decoding with pure CTC
./run_streaming.sh --stage 6 --stop_stage 6 --tag streaming --decode_set test-clean --rnnlm_weight 0.6 --ctc_weight 1.0
```

We obtain the following WER results on test-clean using a model of 12 encoder layers, 6 decoder layers, and attention_dim=512:

|                     |test-clean WER (%)   |
| :-------------------------------| :----:  |
| offline model + offline decode  |  2.4    |
| online model  + offline decode  |  2.9    |
| online model + CTC decode       |  4.0    |
| online model + online decode    |  **3.3**  |

The streaming decoding algorithm is clearly more accurate than pure CTC-based beam search, with improvements coming from the Attention module.

## References

[1] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition". *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017.

[2] Niko Moritz, Takaaki Hori and Jonathan Le Roux, "Streaming Automatic Speech Recognition with the Transformer Model". *ICASSP*, 2020. 