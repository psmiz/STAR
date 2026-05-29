#!/usr/bin/env python3
"""
Combine evaluation results from LM Harness and LAMBADA evaluations.
Outputs a single JSON file with averages at the top.
"""

import argparse
import json
import re
import os
from datetime import datetime
from collections import OrderedDict


def parse_lambada_log(log_path):
    """Parse LAMBADA accuracy from Megatron tasks log file."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    if 'LAMBADA_SKIPPED' in content:
        return None

    # Check for errors
    if 'FileNotFoundError' in content or 'Traceback' in content:
        return None

    # Look for accuracy patterns in the log
    # Primary pattern: "avg accuracy: X.XXXXE-XX" (scientific notation from Megatron)
    # Example: "validation results on LAMBADA | number correct: 1.7730E+03 | total examples: 5.1530E+03 | avg accuracy: 3.4407E-01"
    patterns = [
        r'avg accuracy[:\s]+([0-9.]+[Ee][+-]?[0-9]+)',  # Scientific notation
        r'avg accuracy[:\s]+([0-9.]+)',  # Decimal
        r'Lambada accuracy[:\s]+([0-9.]+)',
        r'accuracy[:\s]+([0-9.]+)%?',
        r'Accuracy[:\s]+([0-9.]+)%?',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            acc = float(match.group(1))
            # Convert to decimal if it looks like a percentage
            if acc > 1:
                acc = acc / 100.0
            return acc

    return None


def parse_lm_harness_results(json_path):
    """Parse LM Harness results JSON file."""
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = {}
    if 'results' in data:
        for task_name, task_results in data['results'].items():
            # Get accuracy (prefer acc_norm, then acc)
            if 'acc_norm,none' in task_results:
                results[task_name] = {
                    'accuracy': task_results['acc_norm,none'],
                    'metric': 'acc_norm'
                }
            elif 'acc,none' in task_results:
                results[task_name] = {
                    'accuracy': task_results['acc,none'],
                    'metric': 'acc'
                }
            elif 'acc_norm' in task_results:
                results[task_name] = {
                    'accuracy': task_results['acc_norm'],
                    'metric': 'acc_norm'
                }
            elif 'acc' in task_results:
                results[task_name] = {
                    'accuracy': task_results['acc'],
                    'metric': 'acc'
                }

    return results


def combine_results(lm_harness_path, lambada_log_path, checkpoint, iteration):
    """Combine all evaluation results into a single structured output."""

    # Parse LM Harness results
    lm_results = parse_lm_harness_results(lm_harness_path)

    # Parse LAMBADA results
    lambada_acc = parse_lambada_log(lambada_log_path)

    # Build task results
    task_results = OrderedDict()

    # Add LM Harness tasks in alphabetical order
    for task_name in sorted(lm_results.keys()):
        task_results[task_name] = {
            'accuracy': lm_results[task_name]['accuracy'],
            'accuracy_pct': round(lm_results[task_name]['accuracy'] * 100, 2),
            'metric': lm_results[task_name]['metric'],
            'source': 'lm_harness'
        }

    # Add LAMBADA
    if lambada_acc is not None:
        task_results['lambada'] = {
            'accuracy': lambada_acc,
            'accuracy_pct': round(lambada_acc * 100, 2),
            'metric': 'acc',
            'source': 'megatron_tasks'
        }

    # Calculate averages
    all_accuracies = [r['accuracy'] for r in task_results.values()]

    if all_accuracies:
        avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    else:
        avg_accuracy = 0.0

    # Calculate grouped averages
    lm_harness_accs = [r['accuracy'] for r in task_results.values() if r['source'] == 'lm_harness']
    lambada_accs = [r['accuracy'] for r in task_results.values() if r['source'] == 'megatron_tasks']

    lm_harness_avg = sum(lm_harness_accs) / len(lm_harness_accs) if lm_harness_accs else 0.0

    # Build final output with averages at the top
    output = OrderedDict()

    # Summary section at the top
    output['summary'] = OrderedDict([
        ('total_average', round(avg_accuracy * 100, 2)),
        ('total_average_decimal', round(avg_accuracy, 4)),
        ('lm_harness_average', round(lm_harness_avg * 100, 2)),
        ('lambada_accuracy', round(lambada_acc * 100, 2) if lambada_acc else None),
        ('num_tasks', len(task_results)),
    ])

    # Metadata
    output['metadata'] = OrderedDict([
        ('checkpoint', checkpoint),
        ('iteration', iteration),
        ('timestamp', datetime.now().isoformat()),
        ('tasks_evaluated', list(task_results.keys())),
    ])

    # Individual task results
    output['results'] = task_results

    return output


def format_results_display(results):
    """Format results for console display."""
    lines = []
    lines.append("=" * 60)
    lines.append("COMBINED EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    summary = results['summary']
    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(f"  Total Average:        {summary['total_average']:.2f}%")
    lines.append(f"  LM Harness Average:   {summary['lm_harness_average']:.2f}%")
    if summary['lambada_accuracy'] is not None:
        lines.append(f"  LAMBADA Accuracy:     {summary['lambada_accuracy']:.2f}%")
    lines.append(f"  Number of Tasks:      {summary['num_tasks']}")
    lines.append("")

    # Individual results
    lines.append("INDIVIDUAL TASK RESULTS")
    lines.append("-" * 60)
    for task_name, task_result in results['results'].items():
        acc = task_result['accuracy_pct']
        metric = task_result['metric']
        lines.append(f"  {task_name:20s}: {acc:6.2f}% ({metric})")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Combine evaluation results')
    parser.add_argument('--lm-harness-results', type=str, required=True,
                        help='Path to LM Harness results JSON')
    parser.add_argument('--lambada-log', type=str, required=True,
                        help='Path to LAMBADA evaluation log')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save combined results')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Checkpoint path for metadata')
    parser.add_argument('--iteration', type=str, default='',
                        help='Iteration number for metadata')

    args = parser.parse_args()

    # Combine results
    results = combine_results(
        args.lm_harness_results,
        args.lambada_log,
        args.checkpoint,
        args.iteration
    )

    # Save JSON output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print formatted results
    print(format_results_display(results))

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
