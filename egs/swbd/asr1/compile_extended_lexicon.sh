cd /export/home/espnet/egs/swbd/asr1
mkdir -p data/extended_lexicon/
cd data/extended_lexicon/

# May 12, 2020. Weiran: i think it is not necessary to get lexicon from wsj and librispeech.
lexicon_wsj=/export/home/eesen-master/asr_egs/wsj/data/local/dict_phn/lexicon.txt
paste -d\  \
  <(awk '{ print $1 }' ${lexicon_wsj} | tr '[:upper:]' '[:lower:]') \
  <(cut -d\  -f 2- ${lexicon_wsj} | tr '[:upper:]' '[:lower:]' | sed -e 's/0//g' -e 's/1//g' -e 's/2//g' -e 's/3//g' -e 's/4//g' -e 's/<//g' -e 's/>//g') \
| sort > lexicon_from_wsj.txt

lexicon_ls=/export/home/eesen-master/asr_egs/librispeech/config/librispeech_phn_reduced_dict.txt
paste -d\  \
  <(awk '{ print $1 }' ${lexicon_ls} | tr '[:upper:]' '[:lower:]') \
  <(cut -d\  -f 2- ${lexicon_ls} | tr '[:upper:]' '[:lower:]' | sed -e 's/<//g' -e 's/>//g') \
| sort > lexicon_from_librispeech.txt

lexicon_swbd=/export/home/espnet/egs/swbd/asr1/data/local/dict_nosp/lexicon.txt
paste -d\  \
  <(awk '{ print $1 }' ${lexicon_swbd} | tr '[:upper:]' '[:lower:]') \
  <(cut -d\  -f 2- ${lexicon_swbd} | tr '[:upper:]' '[:lower:]' | sed -e 's/<//g' -e 's/>//g') \
| sort > lexicon_from_swbd.txt


cat lexicon_from_wsj.txt lexicon_from_librispeech.txt lexicon_from_swbd.txt | sort | uniq > lexicon_large_tmp

gawk '{ print $1 }' /export/home/espnet/egs/swbd/asr1/data/train_fisher/words_fisher.txt | sort | uniq |\
      grep -v "\." | grep -v "-" | grep -v "'" | grep -v "?" | grep -v "," | grep -v "_" | grep -v "[0-9]" | grep -v "]" | grep -v "&" > words1
gawk '{ print $1 }' lexicon_large_tmp | sort | uniq > words2
comm -23 words1 words2 > words_unknown
cut -d\  -f2- /export/home/espnet/egs/swbd/asr1/data/train_fisher/text | tr " " "\n" | sed '/^$/d' | sort | uniq -c |\
    gawk '{ if ($1>2) print $2 }' | sort | uniq > words_frequent

comm -12 words_unknown words_frequent > words_unknown_frequent

g2p-seq2seq --decode words_unknown_frequent --model_dir ../../g2p-seq2seq/g2p-seq2seq-model-6.2-cmudict-nostress/ --output words_unknown_lexicon.txt
cat words_unknown_lexicon.txt | tr '[:upper:]' '[:lower:]' > words_unknown_lexicon_lower.txt
cat words_unknown_lexicon_lower.txt lexicon_large_tmp | sort > lexicon_large.txt

# Convert the lexicon itself, no bpe yet.
python ../../convert_phone_and_alphabet.py --phonefile ../../data/local/dict_phone/phones.txt --infile lexicon_large.txt \
    --outfile ./lexicon_large_phone_converted.txt --in_separator " " --out_separator " "

################################################################################################################

# May 12, 2020. Weiran: I think we need g2p to convert all words (not just frequent ones)
# into phone sequence so we do not have <unk> phones for training.
gawk '{ print $1 }' data/extended_lexicon/lexicon_large.txt | sort | uniq > known_words
cut -d\  -f 2- data/train_fisher/text_clean | sed 's/\._/\. /g' | tr " " "\n" | sort | uniq |\
perl -lane 'BEGIN{open(A,"known_words"); while(<A>){chomp; $k{$_}++}}
    print $_ if not defined($k{$F[0]}); ' > unknown_words1
cut -d\  -f 2- data/train_nodup/text | sed 's/\._/\. /g' | tr " " "\n" | sort | uniq |\
perl -lane 'BEGIN{open(A,"known_words"); while(<A>){chomp; $k{$_}++}}
    print $_ if not defined($k{$F[0]}); ' > unknown_words2
cat unknown_words1 unknown_words2 | sort | uniq > unknown_words
g2p-seq2seq --decode unknown_words --model_dir g2p-seq2seq/g2p-seq2seq-model-6.2-cmudict-nostress/ --output unknown_words_lexicon.txt

# Combine with order. ONLY keep ONE pronunciation (the swbd one initially).
python combine_lexicons.py --infile1 data/local/dict_nosp/lexicon.txt \
  --infile2 data/extended_lexicon/lexicon_large.txt \
  --infile3 unknown_words_lexicon.txt \
  --outfile lexicon_swbd_fisher.txt
sed -i -e 's:okay m k ey:okay ow k ey:' lexicon_swbd_fisher.txt
echo "ak47 ey k ey f ao r t iy s eh v ih n" >> lexicon_swbd_fisher.txt
echo "ak47s ey k ey f ao r t iy s eh v ih n z" >> lexicon_swbd_fisher.txt
sed -i '/^phd/d' lexicon_swbd_fisher.txt
sed -i '/^ymca/d' lexicon_swbd_fisher.txt
sed -i '/^\. /d' lexicon_swbd_fisher.txt
rm unknown_words*

# Replace some phones.
cat lexicon_swbd_fisher.txt |\
  sed 's/ ax/ ah/g' |\
  sed 's/ en/ ah n/g' |\
  sed 's/ el/ ah l/g' |\
  sort | uniq > lexicon_swbd_fisher_merged.txt

# Convert the lexicon itself, no bpe yet.
python convert_phone_and_alphabet.py --phonefile data/local/dict_phone/phones.txt --infile lexicon_swbd_fisher.txt \
    --outfile lexicon_swbd_fisher_phone_converted.txt --in_separator " " --out_separator " "
sed -i '/\[vocalized-noise\]/d' lexicon_swbd_fisher_phone_converted.txt
# Guangsen said this is what happened for kaldi's swbd+fisher recipe.
# cat $dir/acronyms_lex_swbd.txt |\
# sed ‘s/ ax/ ah/g’ |\
# sed ‘s/ en/ ah n/g’ |\
# sed ‘s/ el/ ah l/g’ \
# > $dir/acronyms_lex_swbd_cmuphones.txt