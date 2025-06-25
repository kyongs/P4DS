#!/usr/bin/env python3
"""
Analyze accuracy by number of tasks to evaluate the effectiveness of query decomposition.
"""

import json
import pandas as pd
from collections import defaultdict
import argparse

def analyze_accuracy_by_tasks(results_file, scores_file):
    """
    Analyze accuracy by number of tasks
    
    Args:
        results_file: Path to results_thoughts_vX.json 
        scores_file: Path to results_with_score_vX.json
    """
    
    # Load data
    print("Loading data...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    
    print(f"Loaded {len(results)} results and {len(scores)} scores")
    
    # Ensure data alignment
    if len(results) != len(scores):
        print(f"Warning: Mismatched lengths - results: {len(results)}, scores: {len(scores)}")
        min_len = min(len(results), len(scores))
        results = results[:min_len]
        scores = scores[:min_len]
        print(f"Using first {min_len} entries")
    
    # Combine data
    combined_data = []
    for i, (result, score) in enumerate(zip(results, scores)):
        combined_data.append({
            'question_id': i,
            'question': result['question'],
            'num_tasks': result['execution_info']['num_tasks'],
            'execution_type': result['execution_info']['execution_type'],
            'needs_decomposition': result['execution_info']['needs_decomposition'],
            'complexity_reason': result['execution_info']['complexity_reason'],
            'system_answer': score['System Answer'],
            'true_answer': score['True Answer'],
            'score': score['Score'],
            'level': score.get('Level', 'Unknown')
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(combined_data)
    
    print("\n" + "="*80)
    print("ACCURACY BY NUMBER OF TASKS")
    print("="*80)
    
    # Group by num_tasks and calculate accuracy
    task_stats = df.groupby('num_tasks').agg({
        'score': ['count', 'sum', 'mean'],
        'needs_decomposition': 'first'  # Should be same for all in group
    }).round(4)
    
    task_stats.columns = ['total_questions', 'correct_answers', 'accuracy', 'needs_decomposition']
    
    print(task_stats)
    
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)
    
    for num_tasks in sorted(df['num_tasks'].unique()):
        subset = df[df['num_tasks'] == num_tasks]
        accuracy = subset['score'].mean()
        total = len(subset)
        correct = subset['score'].sum()
        
        print(f"\n{num_tasks} Task(s):")
        print(f"  Total questions: {total}")
        print(f"  Correct answers: {correct}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Needs decomposition: {subset['needs_decomposition'].iloc[0]}")
        print(f"  Execution types: {subset['execution_type'].value_counts().to_dict()}")
        
        # Show some example questions
        print(f"  Example questions:")
        for i, (_, row) in enumerate(subset.head(3).iterrows()):
            status = "✓" if row['score'] == 1 else "✗"
            print(f"    {status} Q{row['question_id']+1}: {row['question'][:80]}...")
    
    print("\n" + "="*80)
    print("EXECUTION TYPE ANALYSIS")
    print("="*80)
    
    type_stats = df.groupby('execution_type').agg({
        'score': ['count', 'sum', 'mean'],
        'num_tasks': 'mean'
    }).round(4)
    
    type_stats.columns = ['total_questions', 'correct_answers', 'accuracy', 'avg_num_tasks']
    print(type_stats)
    
    print("\n" + "="*80)
    print("DECOMPOSITION vs NO DECOMPOSITION")
    print("="*80)
    
    decomp_stats = df.groupby('needs_decomposition').agg({
        'score': ['count', 'sum', 'mean'],
        'num_tasks': 'mean'
    }).round(4)
    
    decomp_stats.columns = ['total_questions', 'correct_answers', 'accuracy', 'avg_num_tasks']
    print(decomp_stats)
    
    # Simple comparison without statistical test
    decomp_true_acc = df[df['needs_decomposition'] == True]['score'].mean()
    decomp_false_acc = df[df['needs_decomposition'] == False]['score'].mean()
    
    print(f"\nComparison:")
    print(f"  Simple queries (no decomposition): {decomp_false_acc:.4f} accuracy")
    print(f"  Complex queries (with decomposition): {decomp_true_acc:.4f} accuracy")
    print(f"  Difference: {decomp_false_acc - decomp_true_acc:.4f}")
    
    print("\n" + "="*80)
    print("COMPLEXITY REASON ANALYSIS")
    print("="*80)
    
    # Group by complexity reason and show accuracy
    complexity_stats = df.groupby('complexity_reason').agg({
        'score': ['count', 'sum', 'mean'],
        'num_tasks': 'mean'
    }).round(4)
    
    complexity_stats.columns = ['total_questions', 'correct_answers', 'accuracy', 'avg_num_tasks']
    print(complexity_stats)
    
    print("\n" + "="*80)
    print("SUMMARY INSIGHTS")
    print("="*80)
    
    total_accuracy = df['score'].mean()
    print(f"Overall accuracy: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
    
    # Task count distribution
    task_dist = df['num_tasks'].value_counts().sort_index()
    print(f"\nTask count distribution:")
    for tasks, count in task_dist.items():
        percentage = count / len(df) * 100
        print(f"  {tasks} task(s): {count} questions ({percentage:.1f}%)")
    
    # Best and worst performing task counts
    best_task_count = df.groupby('num_tasks')['score'].mean().idxmax()
    worst_task_count = df.groupby('num_tasks')['score'].mean().idxmin()
    
    print(f"\nBest performing task count: {best_task_count} task(s)")
    print(f"Worst performing task count: {worst_task_count} task(s)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze accuracy by number of tasks')
    parser.add_argument('--results', '-r', default='./results/v0609/results_thoughts_v12.json',
                        help='Path to results file')
    parser.add_argument('--scores', '-s', default='./scores/v0609/results_with_score_v12.json', 
                        help='Path to scores file')
    parser.add_argument('--output', '-o', help='Save detailed results to CSV file')
    
    args = parser.parse_args()
    
    # Run analysis
    df = analyze_accuracy_by_tasks(args.results, args.scores)
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main() 