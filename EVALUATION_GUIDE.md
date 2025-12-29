# Evaluation Guide: Running Evaluation and Comparing Results

This guide explains how to run evaluation after training is complete and compare results across different methods.

## Overview

After training models, you can evaluate them using various evaluation scripts. All results are saved to CSV files in the `./csvs/` directory, which makes it easy to compare different methods.

## Step 1: Verify Training is Complete

Before running evaluation, ensure your models are saved in the checkpoint directory specified in your config file. For example, if using `cifar5_resnet20.py`:

- Config location: `configs/cifar5_resnet20.py`
- Checkpoint directory: `./checkpoints/cifar5_traincliphead/` (as specified in the config)
- Model files should be saved as: `{model_name}_v0.pth.tar` in subdirectories

## Step 2: Choose Your Evaluation Script

The evaluation scripts are located in:
- **Non-ImageNet experiments**: `non_imnet_evaluation_scripts/`
- **ImageNet experiments**: `imnet_evaluation_scripts/` or `evaluation_scripts/`

### Available Evaluation Scripts

1. **`base_model_concept_merging.py`**: Evaluates individual pretrained models and their ensemble
   - Evaluates each base model separately
   - Evaluates the ensemble of all models
   - Output: `base_models.csv`

2. **`auxiliary_concept_merging.py`**: Evaluates baseline merging methods (permute, identity, etc.)
   - Tests various baseline matching functions
   - Output: `auxiliary_functions.csv`

3. **`zipit_concept_merging.py`**: Runs ZipIt! merging with specified hyperparameters
   - Uses best hyperparameters from hyperparameter search
   - Output: `zipit_configurations.csv`

4. **`hyperparameter_testing.py`**: Performs grid search to find best hyperparameters
   - Tests different hyperparameter combinations
   - Output: `hyperparameter_search.csv`

5. **`evaluate_gitrebasin.py`**: Evaluates models merged using git-rebasin
   - Requires pre-merged models from git-rebasin
   - Output: `gitrebasin.csv`

6. **`multidataset_merging.py`**: Evaluates multi-dataset merging experiments
   - For experiments with multiple datasets
   - Output: `multidataset_merging.csv`

## Step 3: Configure the Evaluation Script

Each evaluation script has a few key parameters you need to set:

1. **`config_name`**: The name of your config file (without `.py` extension)
   ```python
   config_name = 'cifar5_resnet20'  # Change this to your config
   ```

2. **`skip_pair_idxs`**: Pairs to skip (optional)
   ```python
   skip_pair_idxs = [0]  # Skip the first pair
   ```

3. **For `zipit_concept_merging.py`**: Set experiment configurations
   ```python
   experiment_configs = [
       {'stop_node': 21, 'params':{'a': .0001, 'b': .075}},
       # Add more configurations as needed
   ]
   ```

4. **For `auxiliary_concept_merging.py`**: Set merging functions to test
   ```python
   merging_fns = [
       'match_tensors_permute',
       'match_tensors_identity',
   ]
   ```

## Step 4: Run the Evaluation

### For Non-ImageNet Experiments:
```bash
# Base models evaluation
python -m non_imnet_evaluation_scripts.base_model_concept_merging

# Auxiliary/baseline methods
python -m non_imnet_evaluation_scripts.auxiliary_concept_merging

# ZipIt! evaluation
python -m non_imnet_evaluation_scripts.zipit_concept_merging

# Hyperparameter search
python -m non_imnet_evaluation_scripts.hyperparameter_testing
```

### For ImageNet Experiments:
```bash
# Use scripts from imnet_evaluation_scripts/ or evaluation_scripts/
python -m imnet_evaluation_scripts.imnet200_ab
python -m imnet_evaluation_scripts.imnet200_hyperparam_search
```

## Step 5: Find Your Results

All results are saved to CSV files in the following structure:
```
./csvs/
  {dataset_name}/
    {model_name}/
      {eval_type}/
        {script_name}.csv
```

For example:
```
./csvs/
  cifar5/
    resnet20x4/
      clip/
        base_models.csv
        auxiliary_functions.csv
        zipit_configurations.csv
        hyperparameter_search.csv
```

## Step 6: Compare Results

### Method 1: Using Python/Pandas

Create a comparison script:

```python
import pandas as pd
import os

# Load results from different methods
base_models = pd.read_csv('./csvs/cifar5/resnet20x4/clip/base_models.csv')
auxiliary = pd.read_csv('./csvs/cifar5/resnet20x4/clip/auxiliary_functions.csv')
zipit = pd.read_csv('./csvs/cifar5/resnet20x4/clip/zipit_configurations.csv')

# Compare Joint accuracy
print("Joint Accuracy Comparison:")
print(f"Base Models (Ensemble): {base_models[base_models['Model'] == 'Ensemble']['Joint'].values[0]:.4f}")
print(f"Best Auxiliary: {auxiliary['Joint'].max():.4f}")
print(f"Best ZipIt: {zipit['Joint'].max():.4f}")

# Compare Per-Task Average
print("\nPer-Task Average Comparison:")
print(f"Base Models (Ensemble): {base_models[base_models['Model'] == 'Ensemble']['Per Task Avg'].values[0]:.4f}")
print(f"Best Auxiliary: {auxiliary['Per Task Avg'].max():.4f}")
print(f"Best ZipIt: {zipit['Per Task Avg'].max():.4f}")

# Show detailed comparison
comparison = pd.DataFrame({
    'Method': ['Ensemble', 'Best Auxiliary', 'Best ZipIt'],
    'Joint': [
        base_models[base_models['Model'] == 'Ensemble']['Joint'].values[0],
        auxiliary['Joint'].max(),
        zipit['Joint'].max()
    ],
    'Per-Task Avg': [
        base_models[base_models['Model'] == 'Ensemble']['Per Task Avg'].values[0],
        auxiliary['Per Task Avg'].max(),
        zipit['Per Task Avg'].max()
    ]
})
print("\nDetailed Comparison:")
print(comparison)
```

### Method 2: Using Command Line Tools

```bash
# View CSV files
cat ./csvs/cifar5/resnet20x4/clip/base_models.csv
cat ./csvs/cifar5/resnet20x4/clip/zipit_configurations.csv

# Compare specific columns (requires csvkit or similar)
csvcut -c Joint,"Per Task Avg" ./csvs/cifar5/resnet20x4/clip/*.csv
```

### Method 3: Using Excel/Google Sheets

Simply open the CSV files in Excel or Google Sheets and create comparison tables or charts.

## Understanding the Results

Each CSV file contains the following metrics:

- **Joint**: Overall accuracy across all tasks
- **Per Task Avg**: Average accuracy across individual tasks
- **Task {task_name}**: Accuracy for each specific task
- **Time**: Computation time (for merging methods)
- **Merging Fn**: The merging function used
- **Model Name**: The model architecture
- **Split {task}**: The class splits used for each task

## Example Workflow

Here's a complete example for evaluating CIFAR5 ResNet20 models:

```bash
# 1. First, evaluate base models
python -m non_imnet_evaluation_scripts.base_model_concept_merging

# 2. Run hyperparameter search to find best ZipIt! parameters
python -m non_imnet_evaluation_scripts.hyperparameter_testing

# 3. Run ZipIt! with best hyperparameters (update zipit_concept_merging.py with best params)
python -m non_imnet_evaluation_scripts.zipit_concept_merging

# 4. Run baseline methods for comparison
python -m non_imnet_evaluation_scripts.auxiliary_concept_merging

# 5. Compare results
python compare_results.py  # Your custom comparison script
```

## Troubleshooting

1. **Models not found**: Check that the checkpoint directory in your config matches where models were saved during training.

2. **CSV directory doesn't exist**: The scripts automatically create the directory structure, but ensure you have write permissions.

3. **CUDA out of memory**: Reduce batch size in the config or use CPU mode by setting `device = 'cpu'` in the evaluation script.

4. **No pairs found**: Ensure your checkpoint directory structure matches the expected format (models in subdirectories named by class splits).

## Tips

- Run `base_model_concept_merging.py` first to establish baseline performance
- Use `hyperparameter_testing.py` to find optimal parameters before running ZipIt!
- Compare results across multiple runs by keeping different CSV files or adding timestamps
- The ensemble baseline (from base_model_concept_merging.py) provides an upper bound for merging performance

