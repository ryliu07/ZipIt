#!/usr/bin/env python3
"""
Script to compare evaluation results from different methods.
Usage: python compare_results.py <dataset_name> <model_name> <eval_type>
Example: python compare_results.py cifar5 resnet20x4 clip
"""

import pandas as pd
import os
import sys
from pathlib import Path


def load_csv_if_exists(csv_path):
    """Load CSV file if it exists, return None otherwise."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def compare_results(dataset_name, model_name, eval_type):
    """Compare results from different evaluation methods."""
    base_dir = Path('./csvs') / dataset_name / model_name / eval_type
    
    if not base_dir.exists():
        print(f"Error: Results directory not found: {base_dir}")
        print("Make sure you've run evaluation scripts first.")
        return
    
    print(f"\n{'='*60}")
    print(f"Comparing Results: {dataset_name} / {model_name} / {eval_type}")
    print(f"{'='*60}\n")
    
    # Load all available CSV files
    base_models = load_csv_if_exists(base_dir / 'base_models.csv')
    auxiliary = load_csv_if_exists(base_dir / 'auxiliary_functions.csv')
    zipit = load_csv_if_exists(base_dir / 'zipit_configurations.csv')
    hyperparams = load_csv_if_exists(base_dir / 'hyperparameter_search.csv')
    gitrebasin = load_csv_if_exists(base_dir / 'gitrebasin.csv')
    multidataset = load_csv_if_exists(base_dir / 'multidataset_merging.csv')
    
    results = []
    
    # Base models
    if base_models is not None:
        ensemble_row = base_models[base_models.get('Model', pd.Series()) == 'Ensemble']
        if not ensemble_row.empty:
            results.append({
                'Method': 'Ensemble (Upper Bound)',
                'Joint': ensemble_row['Joint'].values[0] if 'Joint' in ensemble_row.columns else 'N/A',
                'Per Task Avg': ensemble_row['Per Task Avg'].values[0] if 'Per Task Avg' in ensemble_row.columns else 'N/A',
            })
        
        # Individual base models
        for idx, row in base_models.iterrows():
            if row.get('Model', '') != 'Ensemble':
                results.append({
                    'Method': f"Base Model: {row.get('Model', 'Unknown')}",
                    'Joint': row.get('Joint', 'N/A'),
                    'Per Task Avg': row.get('Per Task Avg', 'N/A'),
                })
    
    # Auxiliary methods
    if auxiliary is not None:
        best_aux = auxiliary.loc[auxiliary['Joint'].idxmax()] if 'Joint' in auxiliary.columns else None
        if best_aux is not None:
            results.append({
                'Method': f"Best Auxiliary ({best_aux.get('Merging Fn', 'Unknown')})",
                'Joint': best_aux.get('Joint', 'N/A'),
                'Per Task Avg': best_aux.get('Per Task Avg', 'N/A'),
            })
    
    # ZipIt!
    if zipit is not None:
        best_zipit = zipit.loc[zipit['Joint'].idxmax()] if 'Joint' in zipit.columns else None
        if best_zipit is not None:
            results.append({
                'Method': 'Best ZipIt!',
                'Joint': best_zipit.get('Joint', 'N/A'),
                'Per Task Avg': best_zipit.get('Per Task Avg', 'N/A'),
            })
    
    # Git-rebasin
    if gitrebasin is not None:
        best_rebasin = gitrebasin.loc[gitrebasin['Joint'].idxmax()] if 'Joint' in gitrebasin.columns else None
        if best_rebasin is not None:
            results.append({
                'Method': 'Git-rebasin',
                'Joint': best_rebasin.get('Joint', 'N/A'),
                'Per Task Avg': best_rebasin.get('Per Task Avg', 'N/A'),
            })
    
    # Multi-dataset
    if multidataset is not None:
        best_multi = multidataset.loc[multidataset['Joint'].idxmax()] if 'Joint' in multidataset.columns else None
        if best_multi is not None:
            results.append({
                'Method': 'Best Multi-dataset',
                'Joint': best_multi.get('Joint', 'N/A'),
                'Per Task Avg': best_multi.get('Per Task Avg', 'N/A'),
            })
    
    if not results:
        print("No results found. Make sure evaluation scripts have been run.")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Print comparison
    print("Method Comparison:")
    print("-" * 60)
    for _, row in comparison_df.iterrows():
        joint = row['Joint']
        per_task = row['Per Task Avg']
        if isinstance(joint, (int, float)):
            joint_str = f"{joint:.4f}"
        else:
            joint_str = str(joint)
        if isinstance(per_task, (int, float)):
            per_task_str = f"{per_task:.4f}"
        else:
            per_task_str = str(per_task)
        print(f"{row['Method']:30s} | Joint: {joint_str:8s} | Per-Task Avg: {per_task_str:8s}")
    print("-" * 60)
    
    # Find best method
    numeric_results = comparison_df.copy()
    numeric_results['Joint'] = pd.to_numeric(numeric_results['Joint'], errors='coerce')
    numeric_results['Per Task Avg'] = pd.to_numeric(numeric_results['Per Task Avg'], errors='coerce')
    
    if not numeric_results['Joint'].isna().all():
        best_joint = numeric_results.loc[numeric_results['Joint'].idxmax()]
        print(f"\nBest Joint Accuracy: {best_joint['Method']} ({best_joint['Joint']:.4f})")
    
    if not numeric_results['Per Task Avg'].isna().all():
        best_per_task = numeric_results.loc[numeric_results['Per Task Avg'].idxmax()]
        print(f"Best Per-Task Avg: {best_per_task['Method']} ({best_per_task['Per Task Avg']:.4f})")
    
    # Save comparison to CSV
    output_path = base_dir / 'comparison.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")
    
    # Print detailed task-specific results if available
    if zipit is not None and len(zipit) > 0:
        print("\n" + "="*60)
        print("Detailed ZipIt! Results (Top 5 by Joint Accuracy):")
        print("="*60)
        top_zipit = zipit.nlargest(5, 'Joint') if 'Joint' in zipit.columns else zipit.head(5)
        for col in ['Joint', 'Per Task Avg', 'Merging Fn', 'Time']:
            if col in top_zipit.columns:
                print(f"\n{col}:")
                print(top_zipit[col].to_string())


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compare_results.py <dataset_name> <model_name> <eval_type>")
        print("Example: python compare_results.py cifar5 resnet20x4 clip")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    eval_type = sys.argv[3]
    
    compare_results(dataset_name, model_name, eval_type)

