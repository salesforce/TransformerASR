MAIN_ROOT=$PWD/../../..

export LC_ALL=C

export PATH=$MAIN_ROOT/tools/kaldi_tools/sctk/bin:$PWD/utils/:$PWD:$PATH
if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $MAIN_ROOT/tools/venv/bin/activate
fi
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export WARP_CTC_PATH=$MAIN_ROOT/tools/warp-ctc/pytorch_binding/build
export LD_LIBRARY_PATH=$WARP_CTC_PATH:$LD_LIBRARY_PATH
