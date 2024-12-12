# 1. LLM-driven multimodal target volume contouring in radiation oncology (NC) (contouring)

https://www.nature.com/articles/s41467-024-53387-y

https://github.com/tvseg/MM-LLM-RO


1. Environment setting

```bash
git clone https://github.com/tvseg/MM-LLM-RO.git
pip install -r requirements.txt
```

2. Dataset

```bash
cd ./dataset/external1
download sample dataset from https://1drv.ms/u/s!AhwNodepZ41oi5c2-gC9wn104Db6UQ?e=geDlPs
unzip sample.zip
cd ..
```

3. Model checkpoints

```bash
cd model/llama2
# You need to install git-lfs to download the model
git lfs install
# You need to get the permission to download the model
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
cd ..
cd ckpt/multimodal
download model_best.pt from https://1drv.ms/u/s!AhwNodepZ41oi5cpB9lo5U5CbXJz1A?e=tsfaHr
cd ..
```

4. Training

```bash
python main.py --logdir ./ckpt/multimodal --save_checkpoint True --max_epochs 10
```

5. Inference

```bash
python main.py --pretrained_dir ./ckpt/multimodal --context True --n_prompts 2 --context_length 8 --test_mode 2 --batch_size 1 --save_interval 10
```