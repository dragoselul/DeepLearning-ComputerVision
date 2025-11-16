"""
Ablation Study: Effect of number of clicks on segmentation performance
Compares weakly supervised vs fully supervised performance
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys
import json
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# Add project directory to path
sys.path.insert(0, project_dir)

from models.UNetModel import UNet
from weak_supervision.weak_dataset import WeakPH2Dataset
from weak_supervision.point_losses import PointSupervisionWithRegularizationLoss
from weak_supervision.train_weak import train_weak_model, evaluate_weak_model
from dataset.PH2Dataset import PH2Dataset
from measure import evaluate_dataset


def run_ablation_study():
    """
    Run complete ablation study with different numbers of clicks
    """
    print("="*80)
    print("WEAK SUPERVISION ABLATION STUDY")
    print("="*80)
    
    # Configuration
    size = (256, 256)
    batch_size = 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 50
    patience = 10
    learning_rate = 1e-4
    
    print(f"\nConfiguration:")
    print(f"  Image size: {size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Max epochs: {epochs}")
    print(f"  Patience: {patience}")
    print(f"  Learning rate: {learning_rate}")
    
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
    
    # Ablation configurations
    click_configs = [
        (1, 1, 'centroid'),   # Minimal: 1 pos + 1 neg
        (3, 3, 'mixed'),      # Few: 3 pos + 3 neg
        (5, 5, 'mixed'),      # Medium: 5 pos + 5 neg
        (10, 10, 'mixed'),    # Many: 10 pos + 10 neg
        (15, 15, 'mixed'),    # Lots: 15 pos + 15 neg
        (20, 20, 'mixed'),    # Very many: 20 pos + 20 neg
    ]
    
    sampling_strategies = ['random', 'centroid', 'boundary', 'mixed']
    
    results = []
    
    # ===== Part 1: Ablation over number of clicks =====
    print("\n" + "="*80)
    print("PART 1: Ablation over Number of Clicks (Mixed Strategy)")
    print("="*80)
    
    for num_pos, num_neg, strategy in click_configs:
        print(f"\n{'='*80}")
        print(f"Training with {num_pos} positive + {num_neg} negative clicks ({strategy} strategy)")
        print(f"{'='*80}")
        
        # Create datasets
        train_dataset = WeakPH2Dataset(
            split='train',
            transform=transform,
            label_transform=label_transform,
            num_positive=num_pos,
            num_negative=num_neg,
            sampling_strategy=strategy,
            augment=True
        )
        
        val_dataset = WeakPH2Dataset(
            split='val',
            transform=transform,
            label_transform=label_transform,
            num_positive=num_pos,
            num_negative=num_neg,
            sampling_strategy=strategy,
            augment=False
        )
        
        test_dataset = WeakPH2Dataset(
            split='test',
            transform=transform,
            label_transform=label_transform,
            num_positive=num_pos,
            num_negative=num_neg,
            sampling_strategy=strategy,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=2)
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
        # Train model
        model = UNet().to(device)
        loss_fn = PointSupervisionWithRegularizationLoss(
            point_weight=1.0,
            reg_weight=0.01,
            reg_type='tv'
        )
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Save model in script directory
        save_name = os.path.join(script_dir, f'weak_unet_pos{num_pos}_neg{num_neg}_{strategy}.pth')

        trained_model = train_weak_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            opt=opt,
            device=device,
            save_name=save_name,
            epochs=epochs,
            patience=patience
        )
        
        # Predictions directory in script folder
        pred_dir = os.path.join(script_dir, f'weak_predictions/pos{num_pos}_neg{num_neg}_{strategy}')

        # Evaluate on test set with train_weak evaluation
        test_metrics = evaluate_weak_model(
            model_path=save_name,
            test_loader=test_loader,
            device=device,
            output_dir=pred_dir
        )

        # Also evaluate with measure.py for consistency
        gt_files = test_dataset.label_paths
        print(f"\n{'='*60}")
        print(f"Evaluating with measure.py")
        print(f"{'='*60}")
        evaluate_dataset(
            pred_dir=pred_dir,
            gt_files=gt_files,
            dataset_name=f'ph2_weak_pos{num_pos}_neg{num_neg}',
            model_name=strategy,
            pred_pattern='*.png'
        )
        
        # Store results (convert numpy types to Python types for JSON)
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
            'model_path': save_name
        })
    
    # ===== Part 2: Ablation over sampling strategies (5+5 clicks) =====
    print("\n" + "="*80)
    print("PART 2: Ablation over Sampling Strategies (5+5 clicks)")
    print("="*80)
    
    for strategy in sampling_strategies:
        print(f"\n{'='*80}")
        print(f"Training with 5+5 clicks ({strategy} strategy)")
        print(f"{'='*80}")
        
        # Create datasets
        train_dataset = WeakPH2Dataset(
            split='train',
            transform=transform,
            label_transform=label_transform,
            num_positive=5,
            num_negative=5,
            sampling_strategy=strategy,
            augment=True
        )
        
        val_dataset = WeakPH2Dataset(
            split='val',
            transform=transform,
            label_transform=label_transform,
            num_positive=5,
            num_negative=5,
            sampling_strategy=strategy,
            augment=False
        )
        
        test_dataset = WeakPH2Dataset(
            split='test',
            transform=transform,
            label_transform=label_transform,
            num_positive=5,
            num_negative=5,
            sampling_strategy=strategy,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=2)
        
        # Train model
        model = UNet().to(device)
        loss_fn = PointSupervisionWithRegularizationLoss(
            point_weight=1.0,
            reg_weight=0.01,
            reg_type='tv'
        )
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Save model in script directory
        save_name = os.path.join(script_dir, f'weak_unet_5+5_{strategy}.pth')

        trained_model = train_weak_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            opt=opt,
            device=device,
            save_name=save_name,
            epochs=epochs,
            patience=patience
        )
        
        # Predictions directory in script folder
        pred_dir = os.path.join(script_dir, f'weak_predictions/5+5_{strategy}')

        # Evaluate on test set with train_weak evaluation
        test_metrics = evaluate_weak_model(
            model_path=save_name,
            test_loader=test_loader,
            device=device,
            output_dir=pred_dir
        )

        # Also evaluate with measure.py for consistency
        gt_files = test_dataset.label_paths
        print(f"\n{'='*60}")
        print(f"Evaluating with measure.py")
        print(f"{'='*60}")
        evaluate_dataset(
            pred_dir=pred_dir,
            gt_files=gt_files,
            dataset_name=f'ph2_weak_5+5',
            model_name=strategy,
            pred_pattern='*.png'
        )
        
        # Store results (convert numpy types to Python types for JSON)
        results.append({
            'num_positive': 5,
            'num_negative': 5,
            'total_clicks': 10,
            'strategy': strategy,
            'test_dice': float(test_metrics['dice']),
            'test_iou': float(test_metrics['iou']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_sensitivity': float(test_metrics['sensitivity']),
            'test_specificity': float(test_metrics['specificity']),
            'model_path': save_name
        })
    
    # ===== Save Results =====
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Save to JSON in script directory
    results_dir = os.path.join(script_dir, 'ablation_results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'ablation_study_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Clicks':<10} {'Strategy':<12} {'Dice':<8} {'IoU':<8} {'Acc':<8} {'Sens':<8} {'Spec':<8}")
    print("-"*80)
    
    for r in results:
        clicks = f"{r['num_positive']}+{r['num_negative']}"
        print(f"{clicks:<10} {r['strategy']:<12} "
              f"{r['test_dice']:<8.4f} {r['test_iou']:<8.4f} "
              f"{r['test_accuracy']:<8.4f} {r['test_sensitivity']:<8.4f} "
              f"{r['test_specificity']:<8.4f}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['test_dice'])
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Clicks: {best_result['num_positive']}+{best_result['num_negative']}")
    print(f"Strategy: {best_result['strategy']}")
    print(f"Test Dice: {best_result['test_dice']:.4f}")
    print(f"Test IoU: {best_result['test_iou']:.4f}")
    
    return results


if __name__ == '__main__':

    results = run_ablation_study()
    print("\n✓ Ablation study complete!")

