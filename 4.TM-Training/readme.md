# 4. TM-Training

This directory hosts a module for the training of topic models. It provides two topic modeling wrappers: `LDA-Mallet` and `BERTopic`.

## Main modules

### TMTrainer

It contains classes for training Mallet LDA (`MalletLDATrainer`) and BERTopic (`BERTopicTrainer`) models, grouped under the TMTrainer abstract class. Any new topic model trainer should inherit from TMTrainer and implement the abstract methods train() and infer().

> **IMPORTANT TO RUN LDA-Mallet**
>
> Download the [latest release of Mallet](https://github.com/mimno/Mallet/releases) and place it in the `4.TM-Training/src/train` directory. This can be done using the script `4.TM-Trainingbash_scripts/wget_mallet.sh`.

## Configuration File

The `config.yaml` file contains various parameters used to configure the behavior of the system. Below is an example configuration file and a description of its parameters:

```yaml
ntopics: 20
model_type: MalletLda

dft_config_models:
  mallet:
    # Regular expression for token identification
    token_regexp: [\p{L}\p{N}][\p{L}\p{N}\p{P}]*\p{L}
    # Settings for mallet training and doctopics postprocessing
    alpha: 5
    optimize_interval: 10
    num_threads: 4
    num_iterations: 1000
    doc_topic_thr: 0
    num_iterations_inf: 100
  bertopic: 
    no_below: 1
    no_above: 1
    get_sims: False
    sbert_model: paraphrase-distilroberta-base-v2
    umap_n_components: 5
    umap_n_neighbors: 15
    umap_min_dist: 0.0
    umap_metric: cosine
    hdbscan_min_cluster_size: 10
    hdbscan_metric: euclidean
    hdbscan_cluster_selection_method: eom
    hbdsan_prediction_data: True
```

## Command-Line Arguments

The script can be run with several command-line arguments to override the default configuration values. Here are the arguments that can be used:

- `--config`: Path to the configuration file.
- `--ntopics`: Number of topics to train the model with.
- `--model_type`: Type of model to train (`MalletLda` or `BERTopic`).
- `-s`, `--source`: Path to the input parquet file (required).
- `-o`, `--output`: Path to the output parquet file (required).

## Example Usage

To train a topic model on the output of the previous step, use the following command:

```bash
python train_topic_model.py --config path/to/config.yaml -s path/to/input.parquet -o path/to/output/directory
```
