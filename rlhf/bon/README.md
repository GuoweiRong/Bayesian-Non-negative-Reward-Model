# Best-of-N (BoN)

## Overview

For BoN experiment, we use a 20K subset of the Unified-Feedback dataset, labeled by a gold reward model, to train proxy reward models. BoN sampling is then conducted on a held-out 1K test set, where $N$ responses are generated per prompt. The proxy model ranks these responses, selecting the top responses based on their proxy scores. The selected responses are then evaluated by the gold reward model, and average scores (both proxy and gold) across the test set reflect the quality of responses chosen by the proxy reward model.

Scripts for each step can be found in the `scripts/rlhf/bon`. For the data generation part, please refer to `rlhf/data_generation`.


## Experiment Steps


### Step 1: Train Proxy Reward Model

Using the gold reward-labeled training data, we train a proxy reward model to approximate the scoring behavior of the gold model. This model will later be used to score generated responses.

### Step 2: Generate Samples Using the Policy Model

For each prompt in the 1K test set, the policy model generates $N$ responses (where $N$ varies from 1 to 405). These responses represent a wide range of potential answers from which the proxy model will select the best candidate.

### Step 3: Obtain Proxy Scores for Generated Samples

The trained proxy model is applied to each of the generated responses, assigning a proxy reward score to each response. This score reflects the model’s evaluation of the response quality based on the proxy’s learned preferences.

### Step 4: Select Best-of-N Responses

Using the proxy scores from Step 3, we select the single best response out of the $N$ generated responses for each prompt. The highest-scoring response is chosen as the “best-of-N” according to the proxy model. 

**Note**: To apply ensemble methods: (1) train multiple models in Step 1 using different random seeds; (2) perform inference with each model in Step 3; and (3) aggregate the proxy scores using methods like `min` or `avg` before proceeding to Step 4.

### Step 5: Evaluate Selected Responses with Gold Reward Model

The selected best-of-N responses from Step 4 are then evaluated by the gold reward model to obtain a gold score. This step measures the true quality of the responses chosen by the proxy model in alignment with the gold standard.

### Step 6: Collect Results

For each $N$ value (from 1 to 405), we calculate the average proxy and gold scores across all test prompts. These scores reflect the proxy model’s effectiveness at selecting responses that align with gold standards as $N$ increases.
