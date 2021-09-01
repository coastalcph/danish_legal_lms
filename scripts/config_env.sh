conda create -n danish_lm python=3.9
source ~/miniconda3/etc/profile.d/conda.sh
conda activate danish_lm
pip3 install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
