# ⭐ TWIN - Same or Not? Enhancing Visual Perception in Vision-Language Models

This is the code for the paper [Same or Not? Enhancing Visual Perception in Vision-Language Models](https://glab-caltech.github.io/twin/) by [Damiano Marsili](https://damianomarsili.github.io/), [Aditya Mehta](https://aditya-mehta1.github.io/), [Ryan Lin](https://rlin232.github.io/), and [Georgia Gkioxari](https://georgiagkioxari.com/).

<div align="center">
  <a href=""><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://glab-caltech.github.io/twin/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/datasets/glab-caltech/TWIN'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
  <a href='https://huggingface.co/datasets/glab-caltech/FGVQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-blue'></a>
  <a href='https://huggingface.co/collections/glab-caltech/twin'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-purple'></a>

</div>

![image](docs/twin_dataset.png)

## 🚀 Quickstart
Clone the repo:
```bash
git clone --recurse-submodules https://github.com/damianomarsili/TWIN.git
```

We use `uv` to manage all dependencies. If your system does not have `uv`, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Setup your environment:
```bash
cd TWIN
uv sync
```
For post-training with `verl`, you must also install verl's dependencies. You can do so by running the following script:
```
bash modules/verl/scripts/uv_install_vllm_sglang_mcore.sh
```

⚠️ Note: This setup assumes CUDA 12.2 and Python 3.10.  If you are using a different CUDA version, you may need to install a version of `torch` and `flash-attn` compatible with your system.

## 🤗 The TWIN Dataset & FGVQA Benchmark Suite. 
The [TWIN dataset](https://huggingface.co/datasets/glab-caltech/TWIN) and [FGVQA benchmark suite](https://huggingface.co/datasets/glab-caltech/FGVQA) are hosted on Huggingface 🤗.

The dataset and benchmark suite can be accessed with the following code:
```python
from datasets import load_dataset

# TWIN Dataset
twin_dataset = load_dataset("glab-caltech/TWIN")

# FGVQA Benchmark Suite
fgvqa_benchmark = load_dataset("glab-caltech/FGVQA")
```

## 📊 Evaluating on FGVQA
We use [LMMs-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for all evaluations. Model checkpoints post-trained on TWIN are hosted on [Huggingface 🤗](https://huggingface.co/collections/glab-caltech/twin). 

To evaluate a model on FGVQA, please run the following code:
```
bash evals/eval.sh
```

Inside `evals/eval.sh`, you can edit the model checkpoint to evaluate by changing the `MODEL_DIR` and `MODEL_ARGS` variables in the script. To evaluate on a subset of the benchmark suite (e.g. CUB), you can edit the `TASKS` variable to be `fgvqa_{subset}` (e.g. `fgvqa_cub`).

## 🧠 Post-training on TWIN
We use [verl](https://github.com/volcengine/verl) for RL post-training. Prior to training, you must download and preprocess the TWIN dataset. You can do so by running the following script:
```
bash training/prepare_training_data.sh
```

Then, you can launch training via the following command:
```
bash training/run_grpo.sh
```

The trained checkpoint will default save to `training/data/checkpoints/`. You can edit this target directory at the top of the bash script `training/run_grpo.sh`.

## 📚 Citation
If you use the TWIN dataset or FGVQA benchmark suite in your research, please consider citing our work:
```bibtex
TODO.
```