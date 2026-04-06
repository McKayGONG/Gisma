"""
Individual GED value weighted training for GREED model
Per-GED-value weighted sampling training - resampled every epoch
GED 0-10: weight 1.0 per value
GED 11-20: weight 0.9 per value
GED 21-30: weight 0.8 per value
GED 31-40: weight 0.7 per value
GED 41-50: weight 0.6 per value
GED 51-60: weight 0.5 per value
GED 61-70: weight 0.4 per value
GED 71-80: weight 0.3 per value
GED 81-90: weight 0.2 per value
GED 91-100: weight 0.1 per value
GED 101+: shared weight 1.0
"""

import fix_openmp
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
import argparse
import os
from tqdm.auto import tqdm
import time
from collections import Counter

from models import NormGEDModel
import config
from gisma_dataset import load_graph_database, collate_fn

class IndividualGEDWeightSampler(Sampler):
    """
    Individual GED value weight sampler - each GED value has its own weight
    GED 0-10: weight 1.0 per value
    GED 11-20: weight 0.9 per value
    GED 21-30: weight 0.8 per value
    ...
    GED 91-100: weight 0.1 per value
    GED 101+: shared weight 1.0
    """
    def __init__(self, ged_values, max_samples, seed=42):
        self.ged_values = np.array(ged_values)
        self.max_samples = max_samples
        self.seed = seed
        self.epoch = 0
        
        # Assign weight for each GED value
        self.ged_weights = {}
        for ged_val in range(0, 11):      # 0-10: weight 1.0
            self.ged_weights[ged_val] = 1.0
        for ged_val in range(11, 21):     # 11-20: weight 0.9
            self.ged_weights[ged_val] = 0.9
        for ged_val in range(21, 31):     # 21-30: weight 0.8
            self.ged_weights[ged_val] = 0.8
        for ged_val in range(31, 41):     # 31-40: weight 0.7
            self.ged_weights[ged_val] = 0.7
        for ged_val in range(41, 51):     # 41-50: weight 0.6
            self.ged_weights[ged_val] = 0.6
        for ged_val in range(51, 61):     # 51-60: weight 0.5
            self.ged_weights[ged_val] = 0.5
        for ged_val in range(61, 71):     # 61-70: weight 0.4
            self.ged_weights[ged_val] = 0.4
        for ged_val in range(71, 81):     # 71-80: weight 0.3
            self.ged_weights[ged_val] = 0.3
        for ged_val in range(81, 91):     # 81-90: weight 0.2
            self.ged_weights[ged_val] = 0.2
        for ged_val in range(91, 101):    # 91-100: weight 0.1
            self.ged_weights[ged_val] = 0.1
        
        # Group sample indices by GED value
        self.value_indices = {}
        for idx, ged in enumerate(self.ged_values):
            ged_int = int(ged)  # Convert to integer
            if ged_int not in self.value_indices:
                self.value_indices[ged_int] = []
            self.value_indices[ged_int].append(idx)
        
        # Calculate the number of samples for each GED value
        self._calculate_samples_per_value()

        # Print statistics
        self._print_statistics()
    
    def _calculate_samples_per_value(self):
        """Calculate the number of samples for each GED value"""
        # Calculate total weight
        total_weight = 0.0
        value_weights = {}  # Existing GED values and their weights
        
        for ged_val, indices in self.value_indices.items():
            if ged_val <= 100:
                # GED 0-100: use predefined weights
                weight = self.ged_weights.get(ged_val, 0.1)
            else:
                # GED 101+: shared weight 1.0, evenly distributed
                num_values_over_100 = len([v for v in self.value_indices.keys() if v > 100])
                weight = 1.0 / num_values_over_100 if num_values_over_100 > 0 else 0.1
            
            value_weights[ged_val] = weight
            total_weight += weight
        
        # Calculate the number of samples per GED value
        self.samples_per_value = {}
        total_allocated = 0

        # First compute ideal allocation
        for ged_val, weight in value_weights.items():
            available = len(self.value_indices[ged_val])
            ideal_samples = int(self.max_samples * weight / total_weight)
            actual_samples = min(ideal_samples, available)
            self.samples_per_value[ged_val] = actual_samples
            total_allocated += actual_samples
        
        # If total is insufficient, increase proportionally by weight
        if total_allocated < self.max_samples:
            remaining = self.max_samples - total_allocated
            
            # Sort by weight, prioritize higher-weight values
            sorted_values = sorted(value_weights.items(), key=lambda x: x[1], reverse=True)
            
            for ged_val, weight in sorted_values:
                available = len(self.value_indices[ged_val])
                current = self.samples_per_value[ged_val]
                
                if available > current:
                    can_add = min(remaining, available - current)
                    self.samples_per_value[ged_val] += can_add
                    remaining -= can_add
                    if remaining == 0:
                        break
    
    def _print_statistics(self):
        """Print sampling statistics"""
        print("\n=== Individual GED Weight Sampler ===")
        print("Each GED value has its own weight:")
        print("  GED 0-10: weight 1.0 each")
        print("  GED 11-20: weight 0.9 each")
        print("  GED 21-30: weight 0.8 each")
        print("  ...")
        print("  GED 91-100: weight 0.1 each")
        print("  GED 101+: shared weight 1.0")
        
        # Sampling statistics by range
        ranges = [
            (0, 10, "0-10"),
            (11, 20, "11-20"),
            (21, 30, "21-30"),
            (31, 40, "31-40"),
            (41, 50, "41-50"),
            (51, 60, "51-60"),
            (61, 70, "61-70"),
            (71, 80, "71-80"),
            (81, 90, "81-90"),
            (91, 100, "91-100"),
            (101, float('inf'), "101+")
        ]
        
        print("\nSampling distribution by ranges:")
        total_samples = sum(self.samples_per_value.values())
        
        for min_val, max_val, range_str in ranges:
            range_available = 0
            range_sampled = 0
            
            for ged_val in self.value_indices.keys():
                if min_val <= ged_val <= max_val:
                    range_available += len(self.value_indices[ged_val])
                    range_sampled += self.samples_per_value.get(ged_val, 0)
            
            if range_available > 0:
                percentage = (range_sampled / total_samples) * 100 if total_samples > 0 else 0
                print(f"  {range_str}: {range_sampled}/{range_available} samples ({percentage:.1f}%)")
        
        print(f"\nTotal unique GED values: {len(self.value_indices)}")
        print(f"Total samples per epoch: {total_samples}")
        print("Note: Samples are resampled every epoch with individual GED value weights")
    
    def set_epoch(self, epoch):
        """Set the current epoch for dynamic sampling"""
        self.epoch = epoch
    
    def __iter__(self):
        # Use a different random seed for each epoch
        np.random.seed(self.seed + self.epoch)
        
        # Dynamically sample from each GED value
        sampled_indices = []
        
        for ged_val, num_samples in self.samples_per_value.items():
            if num_samples > 0 and ged_val in self.value_indices:
                available_indices = self.value_indices[ged_val]
                if len(available_indices) > 0:
                    # Dynamic sampling: randomly re-select each time
                    indices = np.random.choice(
                        available_indices,
                        size=min(num_samples, len(available_indices)),
                        replace=False  # No replacement
                    )
                    sampled_indices.extend(indices.tolist())
        
        # Shuffle the order
        np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)
    
    def __len__(self):
        return sum(self.samples_per_value.values())

class GEDDataset(Dataset):
    """GED dataset"""
    def __init__(self, graph_id1_list, graph_id2_list, ged_list, graphs):
        self.graph_id1_list = graph_id1_list
        self.graph_id2_list = graph_id2_list
        self.ged_list = ged_list
        self.graphs = graphs
        
    def __len__(self):
        return len(self.graph_id1_list)
    
    def __getitem__(self, idx):
        g1 = self.graphs[self.graph_id1_list[idx]]
        g2 = self.graphs[self.graph_id2_list[idx]]
        ged = torch.tensor(self.ged_list[idx], dtype=torch.float32)  # Fix: do not wrap in list
        return g1, g2, ged, ged  # lb = ub = exact GED

def train_epoch(model, optimizer, loader, sampler, scheduler, device, epoch):
    """Train one epoch"""
    model.train()
    
    # Set the current epoch for dynamic sampling
    sampler.set_epoch(epoch)
    
    total_loss = 0.0
    num_batches = 0
    
    for g, h, lb, ub in tqdm(loader, desc=f'Training Epoch {epoch+1}', leave=False):
        g = g.to(device)
        h = h.to(device)
        lb = lb.to(device)
        ub = ub.to(device)
        
        # Forward pass
        pred = model(g, h)
        loss = model.criterion(lb, ub, pred)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        
        # Update learning rate after each batch (consistent with original GREED)
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, loader, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for g, h, lb, ub in tqdm(loader, desc='Validation', leave=False):
            g = g.to(device)
            h = h.to(device)
            lb = lb.to(device)
            ub = ub.to(device)
            
            pred = model(g, h)
            loss = model.criterion(lb, ub, pred)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Individual GED Weight GREED Training')
    parser.add_argument('--dataset', type=str, default='AIDS',
                        help='Dataset name (default: AIDS)')
    parser.add_argument('--ged_file', type=str, required=True,
                        help='Path to GED results file')
    parser.add_argument('--max_pairs', type=int, default=100000,
                        help='Maximum number of pairs to use (default: 100000)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension (default: 64)')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='Output dimension (default: 64)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for saving (default: same as dataset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--step_size_up', type=int, default=50,
                        help='Number of training iterations (batches) in the increasing half of a cycle (default: 50)')
    parser.add_argument('--step_size_down', type=int, default=50,
                        help='Number of training iterations (batches) in the decreasing half of a cycle (default: 50)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load GED data
    print(f"\n=== Loading data from {args.ged_file} ===")
    ged_df = pd.read_csv(args.ged_file, header=None, names=['graph_id1', 'graph_id2', 'ged'])
    print(f"Loaded {len(ged_df)} GED pairs")
    
    # Weighted sampling of max_pairs samples (instead of random sampling)
    if len(ged_df) > args.max_pairs:
        # Define weights for each GED value
        ged_weights = {}
        for ged_val in range(0, 11):      # 0-10: weight 1.0
            ged_weights[ged_val] = 1.0
        for ged_val in range(11, 21):     # 11-20: weight 0.9
            ged_weights[ged_val] = 0.9
        for ged_val in range(21, 31):     # 21-30: weight 0.8
            ged_weights[ged_val] = 0.8
        for ged_val in range(31, 41):     # 31-40: weight 0.7
            ged_weights[ged_val] = 0.7
        for ged_val in range(41, 51):     # 41-50: weight 0.6
            ged_weights[ged_val] = 0.6
        for ged_val in range(51, 61):     # 51-60: weight 0.5
            ged_weights[ged_val] = 0.5
        for ged_val in range(61, 71):     # 61-70: weight 0.4
            ged_weights[ged_val] = 0.4
        for ged_val in range(71, 81):     # 71-80: weight 0.3
            ged_weights[ged_val] = 0.3
        for ged_val in range(81, 91):     # 81-90: weight 0.2
            ged_weights[ged_val] = 0.2
        for ged_val in range(91, 101):    # 91-100: weight 0.1
            ged_weights[ged_val] = 0.1
        
        # Group by GED value
        ged_groups = ged_df.groupby('ged')
        
        # Calculate the number of samples per GED value
        total_weight = 0.0
        available_counts = {}
        over_100_count = 0  # Count the number of GED values > 100
        
        for ged_val, group in ged_groups:
            ged_int = int(ged_val)
            available_counts[ged_int] = len(group)
            if ged_int <= 100:
                weight = ged_weights.get(ged_int, 0.1)
                total_weight += weight
            else:
                over_100_count += 1
        
        # Values 101+ share weight 1.0
        if over_100_count > 0:
            total_weight += 1.0
        
        # Calculate the number of samples per GED value
        samples_per_ged = {}
        total_allocated = 0

        # First handle GED values 0-100
        for ged_val, count in available_counts.items():
            if ged_val <= 100:
                weight = ged_weights.get(ged_val, 0.1)
                ideal_samples = int(args.max_pairs * weight / total_weight)
                actual_samples = min(ideal_samples, count)
                samples_per_ged[ged_val] = actual_samples
                total_allocated += actual_samples
        
        # Handle GED values 101+ (shared weight 1.0)
        if over_100_count > 0:
            total_over_100_samples = int(args.max_pairs * 1.0 / total_weight)
            # Evenly distribute among all values 101+
            per_value_samples = total_over_100_samples // over_100_count
            remaining = total_over_100_samples % over_100_count
            
            for ged_val, count in available_counts.items():
                if ged_val > 100:
                    base_samples = per_value_samples
                    if remaining > 0:
                        base_samples += 1
                        remaining -= 1
                    actual_samples = min(base_samples, count)
                    samples_per_ged[ged_val] = actual_samples
                    total_allocated += actual_samples
        
        # If total is insufficient, increase by weight
        if total_allocated < args.max_pairs:
            remaining = args.max_pairs - total_allocated
            sorted_geds = sorted(samples_per_ged.keys(), 
                               key=lambda x: ged_weights.get(x, 0.1) if x <= 100 else 0.1, 
                               reverse=True)
            for ged_val in sorted_geds:
                available = available_counts[ged_val]
                current = samples_per_ged[ged_val]
                if available > current:
                    can_add = min(remaining, available - current)
                    samples_per_ged[ged_val] += can_add
                    remaining -= can_add
                    if remaining == 0:
                        break
        
        # Sample from each GED value group according to computed counts
        sampled_dfs = []
        print(f"\nWeighted sampling {args.max_pairs} pairs:")
        
        # Sampling statistics by range
        range_stats = {
            '0-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-50': 0,
            '51-60': 0, '61-70': 0, '71-80': 0, '81-90': 0, '91-100': 0, '101+': 0
        }
        
        for ged_val, num_samples in sorted(samples_per_ged.items()):
            if num_samples > 0:
                group = ged_groups.get_group(ged_val)
                sampled = group.sample(n=num_samples, random_state=args.seed)
                sampled_dfs.append(sampled)
                
                # Accumulate into corresponding range
                if ged_val <= 10:
                    range_stats['0-10'] += num_samples
                elif ged_val <= 20:
                    range_stats['11-20'] += num_samples
                elif ged_val <= 30:
                    range_stats['21-30'] += num_samples
                elif ged_val <= 40:
                    range_stats['31-40'] += num_samples
                elif ged_val <= 50:
                    range_stats['41-50'] += num_samples
                elif ged_val <= 60:
                    range_stats['51-60'] += num_samples
                elif ged_val <= 70:
                    range_stats['61-70'] += num_samples
                elif ged_val <= 80:
                    range_stats['71-80'] += num_samples
                elif ged_val <= 90:
                    range_stats['81-90'] += num_samples
                elif ged_val <= 100:
                    range_stats['91-100'] += num_samples
                else:
                    range_stats['101+'] += num_samples
        
        # Print range statistics
        for range_name, count in range_stats.items():
            if count > 0:
                print(f"  {range_name}: {count} pairs")
        
        ged_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Total weighted samples: {len(ged_df)} pairs")
    
    # Load graph database
    db_file = f"../../Gisma/datasets/{args.dataset}/db.txt"
    print(f"\nLoading graph database from {db_file}")
    graphs, max_label, feature_dim = load_graph_database(db_file)
    print(f"Loaded {len(graphs)} graphs from database")
    
    # Prepare data
    graph_id1_list = ged_df['graph_id1'].tolist()
    graph_id2_list = ged_df['graph_id2'].tolist()
    ged_list = ged_df['ged'].tolist()
    
    # Create dataset
    dataset = GEDDataset(graph_id1_list, graph_id2_list, ged_list, graphs)
    
    # Split into training/validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    indices = list(range(len(dataset)))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"\nDataset split: {train_size} training, {val_size} validation")
    
    # Get training set GED values for dynamic sampling
    train_ged_values = [ged_list[i] for i in train_indices]
    
    # Create dynamic sampler (using Individual GED Weight)
    train_sampler = IndividualGEDWeightSampler(
        train_ged_values,
        max_samples=len(train_indices),  # Use all training data
        seed=args.seed
    )
    
    # Create training/validation subsets, ensuring sampler indices match the dataset
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders (training set uses subset + custom sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,  # Windows compatibility
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].x.shape[1]
    print(f"\nInput dimension: {input_dim}")
    
    # Create model
    model = NormGEDModel(
        n_layers=args.n_layers,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    
    # Optimizer and scheduler (fully consistent with the original GREED paper)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=0,  # Start from 0 (original setting)
        max_lr=args.lr,
        step_size_up=args.step_size_up,  # Batch-level step count
        step_size_down=args.step_size_down,  # Batch-level step count
        cycle_momentum=False
    )
    
    # Create save directory
    if args.model_name is None or args.model_name == args.dataset:
        save_dir = f"./saved_models/{args.dataset}"
    else:
        save_dir = f"./saved_models/{args.dataset}_{args.model_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training log
    log_file = os.path.join(save_dir, "training_log.txt")
    with open(log_file, 'w') as f:
        f.write("# Individual GED Weight Training Log\n")
        f.write("# Epoch\tTrain_Loss\tVal_Loss\tBest_Val\tEpochs_No_Improve\n")
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\n=== Starting individual GED weight training for {args.epochs} epochs ===")
    print(f"Model: {args.n_layers} layers, hidden_dim={args.hidden_dim}, output_dim={args.output_dim}")
    print(f"Training: batch_size={args.batch_size}, max_lr={args.lr}, weight_decay={args.weight_decay}")
    print(f"Learning rate: CyclicLR (0 to {args.lr}), cycle={args.step_size_up+args.step_size_down} batches")
    print(f"  - Step up: {args.step_size_up} batches")
    print(f"  - Step down: {args.step_size_down} batches")
    print(f"Dynamic sampling: Resampling every epoch with individual GED value weights")
    print(f"Early stopping: patience={args.patience} epochs")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train (pass scheduler for batch-level updates)
        train_loss = train_epoch(model, optimizer, train_loader, train_sampler, scheduler, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Note: scheduler.step() is now called after each batch in train_epoch, not needed here
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, os.path.join(save_dir, 'best_model.pt'))
            
            marker = '*'
        else:
            epochs_no_improve += 1
            marker = ''
        
        # Write log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{train_loss:.6f}\t{val_loss:.6f}\t{best_val_loss:.6f}\t{epochs_no_improve}\t{marker}\n")
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s): "
              f"train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, "
              f"best={best_val_loss:.3f} {marker}")
        
        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': args
    }, os.path.join(save_dir, 'final_model.pt'))
    
    print(f"\n=== Training completed ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    print(f"Training log: {log_file}")

if __name__ == "__main__":
    main()
