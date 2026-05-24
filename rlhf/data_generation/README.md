# Generate Data for RLHF

Obtain Gold Reward for Training & Testing Data.

## **Dataset Selection**
A 20K subset of the Unified-Feedback dataset is used for training, and a 1K test set is reserved for evaluation. See sampling in: `rlhf/data_generation/sample_dataset.py`.

## **Labeling**
Both training and test sets are labeled by the [gold reward model](https://huggingface.co/Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback), a 7B human preference model fine-tuned on the full Unified-Feedback dataset. This provides a gold standard against which the proxy models are compared. See : `rlhf/data_generation/obtain_gold_score.py`. After obtaining the gold score, replace the original scores in dataset with the scores labeled by the gold model.

