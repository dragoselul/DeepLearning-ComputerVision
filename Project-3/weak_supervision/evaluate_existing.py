"""
Evaluate existing trained models without retraining
Use this if you already have trained .pth models
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys
import json
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weak_supervision.weak_dataset import WeakPH2Dataset
from weak_supervision.train_weak import evaluate_weak_model


def evaluate_existing_models():
    """
    Evaluate all existing weak supervision models
    Looks for .pth files matching the naming pattern
    """
    print("="*80)
    print("EVALUATING EXISTING WEAK SUPERVISION MODELS")
    print("="*80)

    # Configuration (must match training config)
    size = (256, 256)
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Image size: {size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    label_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    # Find all weak supervision model files
    # Make sure we're in the Project-3 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Go up from weak_supervision to Project-3

    print(f"\nSearching for models in: {project_dir}")
    os.chdir(project_dir)  # Change to project directory
    print(f"Current directory: {os.getcwd()}")

    # Look for both patterns and deduplicate
    all_models = set()

    # Pattern 1: weak_unet_pos{N}_neg{M}_{strategy}.pth
    pattern1_files = glob.glob('weak_unet_pos*_neg*_*.pth')
    print(f"  Pattern 'weak_unet_pos*_neg*_*.pth': found {len(pattern1_files)} files")
    all_models.update(pattern1_files)

    # Pattern 2: weak_unet_{N}+{M}_{strategy}.pth (alternative format)
    pattern2_files = []
    for f in glob.glob('weak_unet_*.pth'):
        if '+' in f:
            pattern2_files.append(f)
            all_models.add(f)
    print(f"  Pattern 'weak_unet_*+*.pth': found {len(pattern2_files)} files")

    # Convert to sorted list
    all_models = sorted(list(all_models))

    if not all_models:
        print("\n✗ No trained models found!")
        print("\nExpected model naming patterns:")
        print("  - weak_unet_pos{N}_neg{M}_{strategy}.pth")
        print("  - weak_unet_{N}+{M}_{strategy}.pth")
        print("\nLooking in current directory:")
        print(f"  {os.getcwd()}")
        print("\nFiles in directory:")
        pth_files = glob.glob('*.pth')
        if pth_files:
            for f in sorted(pth_files)[:10]:
                print(f"  - {f}")
            if len(pth_files) > 10:
                print(f"  ... and {len(pth_files) - 10} more .pth files")
        else:
            print("  (no .pth files found)")
        return []

    print(f"\nFound {len(all_models)} trained model(s):")
    for model_path in all_models:
        print(f"  - {model_path}")

    results = []

    # Parse and evaluate each model
    for model_path in sorted(all_models):
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*80}")

        # Parse model name to get configuration
        config = parse_model_name(model_path)
        if config is None:
            print(f"  ✗ Could not parse model name, skipping...")
            continue

        num_pos, num_neg, strategy = config
        print(f"  Clicks: {num_pos}+{num_neg}")
        print(f"  Strategy: {strategy}")

        # Create test dataset with matching configuration
        test_dataset = WeakPH2Dataset(
            split='test',
            transform=transform,
            label_transform=label_transform,
            num_positive=num_pos,
            num_negative=num_neg,
            sampling_strategy=strategy,
            augment=False
        )

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2)

        print(f"  Test samples: {len(test_dataset)}")

        # Evaluate
        try:
            test_metrics = evaluate_weak_model(
                model_path=model_path,
                test_loader=test_loader,
                device=device,
                output_dir=None  # Don't save predictions again
            )

            # Store results
            results.append({
                'num_positive': int(num_pos),
                'num_negative': int(num_neg),
                'total_clicks': int(num_pos + num_neg),
                'strategy': strategy,
                'test_dice': float(test_metrics['dice']),
                'test_iou': float(test_metrics['iou']),
                'test_accuracy': float(test_metrics['accuracy']),
                'test_sensitivity': float(test_metrics['sensitivity']),
                'test_specificity': float(test_metrics['specificity']),
                'model_path': model_path
            })

            print(f"  ✓ Evaluation complete!")

        except Exception as e:
            print(f"  ✗ Error evaluating model: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\n✗ No models were successfully evaluated!")
        return []

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    os.makedirs('weak_supervision_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'weak_supervision_results/evaluation_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Clicks':<10} {'Strategy':<12} {'Dice':<8} {'IoU':<8} {'Acc':<8} {'Sens':<8} {'Spec':<8}")
    print("-"*80)

    # Sort by clicks then strategy
    results_sorted = sorted(results, key=lambda x: (x['total_clicks'], x['strategy']))

    for r in results_sorted:
        clicks = f"{r['num_positive']}+{r['num_negative']}"
        print(f"{clicks:<10} {r['strategy']:<12} "
              f"{r['test_dice']:<8.4f} {r['test_iou']:<8.4f} "
              f"{r['test_accuracy']:<8.4f} {r['test_sensitivity']:<8.4f} "
              f"{r['test_specificity']:<8.4f}")

    # Print analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Best overall
    best_result = max(results, key=lambda x: x['test_dice'])
    print(f"\nBest Overall Configuration:")
    print(f"  Clicks: {best_result['num_positive']}+{best_result['num_negative']}")
    print(f"  Strategy: {best_result['strategy']}")
    print(f"  Test Dice: {best_result['test_dice']:.4f}")
    print(f"  Test IoU: {best_result['test_iou']:.4f}")

    # Best by click count (if we have multiple click counts)
    click_counts = sorted(set(r['total_clicks'] for r in results))
    if len(click_counts) > 1:
        print(f"\nBest by Click Count:")
        for count in click_counts:
            count_results = [r for r in results if r['total_clicks'] == count]
            if count_results:
                best = max(count_results, key=lambda x: x['test_dice'])
                print(f"  {count:2d} clicks: Dice={best['test_dice']:.4f} ({best['strategy']})")

    # Best by strategy (if we have multiple strategies)
    strategies = sorted(set(r['strategy'] for r in results))
    if len(strategies) > 1:
        print(f"\nBest by Strategy:")
        for strategy in strategies:
            strat_results = [r for r in results if r['strategy'] == strategy]
            if strat_results:
                best = max(strat_results, key=lambda x: x['test_dice'])
                print(f"  {strategy:12s}: Dice={best['test_dice']:.4f} ({best['num_positive']}+{best['num_negative']} clicks)")

    return results


def parse_model_name(model_path):
    """
    Parse model filename to extract configuration

    Examples:
      weak_unet_pos5_neg5_mixed.pth -> (5, 5, 'mixed')
      weak_unet_5+5_random.pth -> (5, 5, 'random')
    """
    import re

    basename = os.path.basename(model_path)

    # Pattern 1: weak_unet_pos{N}_neg{M}_{strategy}.pth
    match1 = re.match(r'weak_unet_pos(\d+)_neg(\d+)_(\w+)\.pth', basename)
    if match1:
        return int(match1.group(1)), int(match1.group(2)), match1.group(3)

    # Pattern 2: weak_unet_{N}+{M}_{strategy}.pth
    match2 = re.match(r'weak_unet_(\d+)\+(\d+)_(\w+)\.pth', basename)
    if match2:
        return int(match2.group(1)), int(match2.group(2)), match2.group(3)

    return None


def create_plots(results_file):
    """
    Create plots from results JSON file
    """
    import matplotlib.pyplot as plt
    import json

    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)

    with open(results_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results to plot!")
        return

    # Plot 1: Dice vs Number of Clicks (for mixed strategy)
    mixed_results = [r for r in results if r['strategy'] == 'mixed']
    if mixed_results:
        mixed_results = sorted(mixed_results, key=lambda x: x['total_clicks'])
        clicks = [r['total_clicks'] for r in mixed_results]
        dice = [r['test_dice'] for r in mixed_results]

        plt.figure(figsize=(10, 6))
        plt.plot(clicks, dice, 'o-', linewidth=2, markersize=8, label='Weak Supervision')
        plt.axhline(y=0.80, color='r', linestyle='--', label='Full Supervision (estimated)', alpha=0.7)
        plt.xlabel('Total Number of Clicks', fontsize=12)
        plt.ylabel('Dice Score', fontsize=12)
        plt.title('Segmentation Performance vs Annotation Effort', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('weak_supervision_results/dice_vs_clicks.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: dice_vs_clicks.png")
        plt.close()

    # Plot 2: Strategy Comparison (for 10 total clicks, or closest available)
    target_clicks = 10
    strategies = sorted(set(r['strategy'] for r in results))

    strat_results = []
    for strategy in strategies:
        # Get results for this strategy near target clicks
        strat_data = [r for r in results if r['strategy'] == strategy]
        if strat_data:
            # Find closest to target
            closest = min(strat_data, key=lambda x: abs(x['total_clicks'] - target_clicks))
            strat_results.append((strategy, closest['test_dice']))

    if len(strat_results) > 1:
        strategies_plot, dice_plot = zip(*strat_results)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies_plot, dice_plot, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(strategies_plot)])
        plt.xlabel('Sampling Strategy', fontsize=12)
        plt.ylabel('Dice Score', fontsize=12)
        plt.title(f'Effect of Sampling Strategy (~{target_clicks} clicks)', fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, dice_val in zip(bars, dice_plot):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dice_val:.3f}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('weak_supervision_results/strategy_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: strategy_comparison.png")
        plt.close()

    print("\n✓ Plots created in weak_supervision_results/")


if __name__ == '__main__':
    import sys

    # Evaluate existing models
    results = evaluate_existing_models()

    if results:
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE!")
        print("="*80)

        # Get the results file path
        import glob
        results_files = sorted(glob.glob('weak_supervision_results/evaluation_*.json'))
        if results_files:
            latest_results = results_files[-1]

            # Offer to create plots
            print(f"\nResults saved to: {latest_results}")
            print("\nTo create plots, run:")
            print(f"  python3 -c \"from weak_supervision.evaluate_existing import create_plots; create_plots('{latest_results}')\"")

            # Auto-create plots if matplotlib available
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                create_plots(latest_results)
            except ImportError:
                print("\nNote: Install matplotlib to auto-generate plots:")
                print("  pip install matplotlib")
    else:
        print("\n✗ No models evaluated. Make sure you have trained models in the current directory.")

