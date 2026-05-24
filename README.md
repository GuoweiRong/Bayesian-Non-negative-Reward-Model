<h1 align="center">Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling</h1>

<p align="center">
  <strong>Bayesian Non-Negative Reward Model (BNRM)</strong><br>
  Robust, uncertainty-aware reward modeling with disentangled non-negative latent factors.
</p>

<p align="center">
  <a href="https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling">
    <img src="https://img.shields.io/badge/HuggingFace-BNRM%20Models-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>
  <a href="https://arxiv.org/abs/2602.10623">
    <img src="https://img.shields.io/badge/arXiv-2602.10623-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  </a>
  <img src="https://img.shields.io/badge/ICML%202026-Oral-4c6fff?style=for-the-badge" alt="ICML 2026 Oral">
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#released-models">Models</a> •
  <a href="#rewardbench-results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#rlhf-ppo">RLHF/PPO</a> •
  <a href="#citation">Citation</a>
</p>

<a id="overview"></a>
## ✨ Overview

Reward models learned from human preferences are central to aligning large language models (LLMs) via reinforcement learning from human feedback, yet they are often vulnerable to reward hacking due to noisy annotations and systematic biases such as response length or style.

We propose **Bayesian Non-Negative Reward Model (BNRM)**, a principled reward modeling framework that integrates non-negative factor analysis into the Bradley-Terry (BT) preference model.

BNRM represents rewards through a sparse, non-negative latent factor generative process operating at two complementary levels: instance-specific latent variables induce disentangled reward representations, while sparsity over global latent factors acts as an implicit debiasing mechanism that suppresses spurious correlations. Together, this disentanglement-then-debiasing structure enables robust uncertainty-aware reward learning.

To scale BNRM to modern LLMs, we develop an amortized variational inference network conditioned on deep model representations, allowing efficient end-to-end training. Extensive empirical results show that BNRM substantially mitigates reward over-optimization, improves robustness under distribution shifts, and yields more interpretable reward decompositions than strong baselines.

🏆 **BNRM has been accepted as an ICML 2026 Oral paper.**

## 🌟 Highlights

| What BNRM Improves | How |
| --- | --- |
| **Reward hacking mitigation** | Sparse non-negative latent factors suppress spurious preference correlations. |
| **Uncertainty-aware learning** | Amortized variational inference models instance-level reward uncertainty end to end. |
| **Robust generalization** | Stronger performance under distribution shift, limited annotations, and noisy labels. |
| **Interpretability** | Non-negative reward decompositions expose more meaningful latent reward factors. |

<a id="released-models"></a>
## 🤗 Released Models

Our reward models are released through the Hugging Face collection below.

| Resource | Link |
| --- | --- |
| 🤗 Model collection | [GuoweiRong/bayesian-non-negative-reward-modeling](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) |
| Contents | Open-source reward models, checkpoints, and related resources for Bayesian non-negative reward modeling. |

<a id="rewardbench-results"></a>
## 📈 RewardBench Results

Results are reported on [RewardBench](https://github.com/allenai/reward-bench). Best scores within each block are **bold** and second-best scores are <u>underlined</u>. Deltas show the improvement over the corresponding base reward model.

### Gemma-2B-it LoRA [Unified-Feedback](https://huggingface.co/datasets/llm-blender/Unified-Feedback)-40K

| Model | Average | Chat | Chat Hard | Safety | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT | 64.5 | **95.8** | 37.3 | 59.9 | 64.8 |
| [BT-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | **72.5 (+8.0)** | 95.6 (-0.2) | **43.3 (+6.0)** | <u>80.9 (+21.0)</u> | **70.1 (+5.3)** |
| [GRM-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | <u>71.8 (+5.0)</u> | <u>95.7 (+1.6)</u> | <u>41.6 (-0.3)</u> | **81.5 (+12.0)** | <u>68.4 (+6.9)</u> |

### Gemma-2-2B-it LoRA Unified-Feedback-40K

| Model | Average | Chat | Chat Hard | Safety | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT | 75.7 | 96.1 | 50.7 | 80.9 | 75.0 |
| [BT-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | <u>79.7 (+4.0)</u> | <u>97.1 (+1.0)</u> | **56.3 (+5.6)** | <u>85.3 (+4.4)</u> | <u>79.9 (+4.9)</u> |
| [GRM-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | **80.5 (+3.2)** | **97.5 (+1.1)** | <u>54.3 (+4.3)</u> | **86.2 (+1.9)** | **84.1 (+5.6)** |

### Gemma-2B-it LoRA Unified-Feedback-400K

| Model | Average | Chat | Chat Hard | Safety | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT | 68.2 | 95.5 | 38.0 | 73.8 | 65.3 |
| [BT-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | **73.2 (+5.0)** | **96.4 (+0.9)** | <u>41.7 (+3.7)</u> | **81.8 (+8.0)** | **72.9 (+7.6)** |
| [GRM-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | <u>71.3 (+0.5)</u> | <u>95.7 (-2.1)</u> | **42.1 (+0.0)** | <u>80.8 (+2.9)</u> | <u>66.7 (+1.5)</u> |

### Gemma-2-2B-it LoRA Unified-Feedback-400K

| Model | Average | Chat | Chat Hard | Safety | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT | 77.5 | 97.2 | 51.4 | 83.2 | 78.3 |
| [BT-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | **79.5 (+2.0)** | **97.5 (+0.3)** | <u>51.7 (+0.3)</u> | <u>84.9 (+1.7)</u> | **83.8 (+5.5)** |
| [GRM-BNRM](https://huggingface.co/collections/GuoweiRong/bayesian-non-negative-reward-modeling) | <u>79.4 (+1.7)</u> | <u>97.4 (-0.5)</u> | **52.9 (+2.1)** | **85.1 (+0.5)** | <u>82.3 (+4.7)</u> |

### Advantages in Low-Resource and Noisy Settings

A robust reward model should remain reliable when preference annotations are scarce or noisy. Using Gemma-2B-it, BT-BNRM shows clear gains over BT in both low-resource training with 1K-20K Unified-Feedback samples and noisy-label training with 40K samples. The improvements become especially pronounced in the most challenging regimes, showing that BNRM is both data-efficient and noise-tolerant for realistic preference-learning scenarios.

| Few-shot Results | Noise Robustness |
| --- | --- |
| <img src="./fewshot.png" alt="Few-shot Results" width="420"> | <img src="./noise.png" alt="Noise Robustness" width="420"> |

<a id="installation"></a>
## 🛠️ Installation

Create the environment:

```bash
conda create -n bnrm python==3.10 -y
conda activate bnrm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`bench.txt` and `swift.txt` record the environments we used for reward-model evaluation and PPO training. They are provided as reference dependency snapshots.

<a id="training"></a>
## 🚀 Training Reward Models

All scripts are under `scripts/`. Adjust dataset paths, model paths, GPU ids, and output directories before running.

| Target | Command |
| --- | --- |
| BT reward model, LoRA | `bash scripts/train_bt_rm_lora.sh` |
| BT reward model, full fine-tuning | `bash scripts/train_bt_rm_full.sh` |
| GRM reward model, LoRA | `bash scripts/train_grm_lora.sh` |
| GRM reward model, full fine-tuning | `bash scripts/train_grm_full.sh` |
| BT-BNRM / GRM-BNRM, LoRA | `bash scripts/train_method_lora.sh` |
| BT-BNRM, full fine-tuning | `bash scripts/train_method_full.sh` |

<a id="evaluation"></a>
## 📊 Evaluating Reward Models

For benchmark evaluation, this repository expects external benchmark code from [RewardBench](https://github.com/allenai/reward-bench) and [RM-Bench](https://github.com/THU-KEG/RM-Bench).

We thank the authors of RewardBench and RM-Bench for their valuable evaluation frameworks.

| Evaluation | Command |
| --- | --- |
| BT reward model | `bash scripts/eval_bt_rm.sh` |
| GRM reward model | `bash scripts/eval_grm_rm.sh` |
| BNRM, LoRA | `bash scripts/eval_method_rm_lora.sh` |
| BNRM, full fine-tuning | `bash scripts/eval_method_rm_full.sh` |
| RewardBench BT | `bash scripts/rewardbench/BT_RewardBench.sh` |
| RewardBench GRM | `bash scripts/rewardbench/GRM_RewardBench.sh` |
| RewardBench BT-BNRM | `bash scripts/rewardbench/BNBT_RewardBench.sh` |
| RewardBench full BT-BNRM | `bash scripts/rewardbench/BNBT_RewardBenchfull.sh` |
| RM-Bench BT | `bash scripts/RMbench/BT.sh` |
| RM-Bench GRM | `bash scripts/RMbench/GRM_RMbench.sh` |
| RM-Bench BNRM | `bash scripts/RMbench/BNRM.sh` |

<a id="rlhf-ppo"></a>
## 🧪 RLHF / PPO

We include PPO scripts for using BNRM as a reward model in RLHF.

Our PPO training scripts are built around **[ms-swift](https://github.com/modelscope/ms-swift)** and our PPO result evaluation uses **[EvalScope](https://github.com/modelscope/evalscope)**:

| Stage | Command |
| --- | --- |
| PPO training | `bash scripts/rlhf/BNRM_swift_ppo/ms_ppo_script.sh` |
| PPO evaluation | `bash scripts/rlhf/BNRM_swift_ppo/evalscope_evaluation_script.sh` |

## 🎯 Best-of-N

We also provide Best-of-N (BoN) scripts for analyzing how proxy reward models select responses from multiple sampled candidates. See `rlhf/bon/README.md` for the full workflow.

> **Tip:** If the reward models have already been trained, Step 1 can be skipped.

| Step | Purpose | Command |
| --- | --- | --- |
| 1 | Train BT proxy reward model | `bash scripts/rlhf/bon/step1_train_proxy_reward_model_baseline.sh` |
| 1 | Train GRM proxy reward model | `bash scripts/rlhf/bon/step1_train_proxy_reward_model_grm.sh` |
| 2 | Generate candidate responses | `bash scripts/rlhf/bon/step2_generate_samples_vllm.sh` |
| 3 | Score candidates with proxy reward models | `bash scripts/rlhf/bon/step3_obtain_proxy_score.sh` |
| 4 | Select Best-of-N responses | `bash scripts/rlhf/bon/step4_choose_best_of_n.sh` |
| 5 | Evaluate selected responses with the gold reward model | `bash scripts/rlhf/bon/step5_obtain_bon_gold_score.sh` |
| 6 | Collect BoN results | `bash scripts/rlhf/bon/step6_collect.sh` |

## 📁 Repository Structure

```text
reward_models/       Reward model training code
rm_eval/             Reward model evaluation code
scripts/             Training, evaluation, RewardBench, RM-Bench, and RLHF scripts
rlhf/                RLHF, PPO, BoN, and data-generation utilities
requirements.txt     Main dependency file
bench.txt            Reward-model evaluation environment reference
swift.txt            PPO/Swift environment reference
```

## 🙏 Acknowledgements

This codebase is built on top of the excellent **[Generalizable Reward Model](https://github.com/YangRui2015/Generalizable-Reward-Model)**.

This project also builds on the open-source LLM alignment ecosystem, including Hugging Face Transformers, PEFT, Accelerate, TRL/Swift, ms-swift, EvalScope, RewardBench, and RM-Bench. We sincerely thank the maintainers and authors of these projects.

<a id="citation"></a>
## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{duan2026mitigatingrewardhackingrlhf,
      title={Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling}, 
      author={Zhibin Duan and Guowei Rong and Zhuo Li and Bo Chen and Mingyuan Zhou and Dandan Guo},
      year={2026},
      eprint={2602.10623},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.10623}, 
}
```
