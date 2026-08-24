import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import NormActivation
from e3nn.o3 import FullyConnectedTensorProduct, Linear
import math 
from e3nn.math import soft_one_hot_linspace
from voxel_convolution import Convolution
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import random
from torch.distributions.normal import Normal
import glob
import re

SPLIT_FILE = "folder_names.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Docking(nn.Module):
    def __init__(self, dockingFFT=None, plot_freq=10, debug=False):
        super(Docking, self).__init__()
        self.debug = debug
        self.plot_freq = plot_freq
        
        self.normalization = "integral"
        self.kernel_size = 5 
        self.num_radial_basis = (self.kernel_size // 2) + 1
        self.lmax = 1
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)

        # 1. Use a standard trainable Linear layer to handle initial scale adaptively
        self.scalar_input = "5x0e"
        self.input_scaler = o3.Linear(irreps_in=self.scalar_input, irreps_out=self.scalar_input)

        self.irreps_per_layer = "8x0e+8x1o" 
        self.irreps_final_layer = "1x0e"  # Single scalar output per voxel grid space

        self.non_linearity = torch.tanh
        self.norm_activation = NormActivation(self.irreps_per_layer, self.non_linearity)
        self.deltaG_bias = nn.Parameter(torch.tensor(0.0))

        # Convolutional layers
        self.conv1 = Convolution(
            device=device, dtype=torch.float32, irreps_in=self.scalar_input, irreps_out=self.irreps_per_layer,
            irreps_sh=self.irreps_sh, diameter=self.kernel_size, num_radial_basis=self.num_radial_basis, normalization=self.normalization
        )
        self.conv2 = Convolution(
            device=device, dtype=torch.float32, irreps_in=self.irreps_per_layer, irreps_out=self.irreps_per_layer,
            irreps_sh=self.irreps_sh, diameter=self.kernel_size, num_radial_basis=self.num_radial_basis, normalization=self.normalization
        )
        self.conv3 = Convolution(
            device=device, dtype=torch.float32, irreps_in=self.irreps_per_layer, irreps_out=self.irreps_per_layer,
            irreps_sh=self.irreps_sh, diameter=self.kernel_size, num_radial_basis=self.num_radial_basis, normalization=self.normalization
        )
        self.conv4 = Convolution(
            device=device, dtype=torch.float32, irreps_in=self.irreps_per_layer, irreps_out=self.irreps_final_layer,
            irreps_sh=self.irreps_sh, diameter=self.kernel_size, num_radial_basis=self.num_radial_basis, normalization=self.normalization
        )

    def channel_wise_NormActivation(self, volume):
        
        B, C, X, Y, Z = volume.shape
        sub_arr = volume.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        sub_arr_norm_activated = self.norm_activation(sub_arr)
        result = sub_arr_norm_activated.view(B, X, Y, Z, C).permute(0, 4, 1, 2, 3)
        return result

    def forward_features(self, x):
        # x comes in as standard PyTorch shape: [B, C, X, Y, Z] (e.g., [1, 5, 31, 31, 31])
        
        # Move feature channel dimension to the end for e3nn.o3.Linear -> [B, X, Y, Z, C]
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # Now e3nn scaling reads '5' accurately regardless of whether batch size is 1 or 100
        feat_perm = self.input_scaler(x_perm)
        
        # Permute back to standard layout for 3D Convolutions -> [B, C, X, Y, Z]
        feat = feat_perm.permute(0, 4, 1, 2, 3).contiguous()
        
        # Process convolutional features
        feat = self.conv1(feat)
        feat = self.channel_wise_NormActivation(feat)
        feat = self.conv2(feat)
        feat = self.channel_wise_NormActivation(feat)
        feat = self.conv3(feat)
        feat = self.channel_wise_NormActivation(feat)
        feat = self.conv4(feat)
        return feat

    def compute_conformer_energy(self, x):
        """Predicts a single scalar energy value for a structure by summing over the grid."""
        energy_grid = self.forward_features(x) 
        return torch.sum(energy_grid, dim=(1, 2, 3, 4)) 

    def forward(self, complex_batch, protein_batch, peptide_batch):
        """
        Calculates Free Energy using Ensemble Exponential Averaging,
        combined with pose prediction via cross entropy on the energy grid.
        """
        RT = 0.001987204 * 298.15  # kcal/mol at 298.15 K

        # --- 1. Compute energies & retain complex grid for pose prediction ---
        energy_grid_complex = self.forward_features(complex_batch)
        E_complex = torch.sum(energy_grid_complex, dim=(1, 2, 3, 4))

        E_protein = self.compute_conformer_energy(protein_batch)
        E_peptide = self.compute_conformer_energy(peptide_batch)

        # Check if we are assessing an individual structure vs an ensemble
        if complex_batch.shape[0] == 1:
            # Simple linear free energy difference
            delta_G = E_complex - (E_protein + E_peptide)
            delta_G = delta_G.squeeze(0)
        else:
            # Thermal Ensemble Average via logsumexp
            G_complex = -RT * torch.logsumexp(-E_complex / RT, dim=0)
            G_protein = -RT * torch.logsumexp(-E_protein / RT, dim=0)
            G_peptide = -RT * torch.logsumexp(-E_peptide / RT, dim=0)

            delta_G = G_complex - (G_protein + G_peptide)

        # Learnable bias — trained at 10x slower lr via a separate parameter group
        delta_G = delta_G + self.deltaG_bias

        # --- 2. Pose Prediction Logic (Cross Entropy) ---
        batch_size = energy_grid_complex.shape[0]
        batch_flat = energy_grid_complex.reshape(batch_size, -1)

        # Softmin probabilities
        probs = F.softmin(batch_flat, dim=1)
        gt_probs = probs[:, 0]
        pred_indices = torch.argmin(batch_flat, dim=1)
        pred_probs = probs[torch.arange(batch_size), pred_indices]

        # Euclidean distances from predicted voxel to grid center
        coords = torch.stack(
            torch.unravel_index(pred_indices, energy_grid_complex.shape[1:]),
            dim=1
        ).float().to(complex_batch.device)

        target_coord = torch.zeros_like(coords)
        grid_size = energy_grid_complex.shape[-1]

        min_dis = torch.min(
            torch.abs(coords - target_coord),
            torch.abs(coords - target_coord - grid_size)
        )
        euclidean_distances = torch.norm(min_dis, dim=1)

        # Location loss (Cross Entropy) — target is voxel index 0 (grid center)
        logits = -batch_flat
        target_indices = torch.zeros(batch_size, dtype=torch.long, device=complex_batch.device)
        location_loss_fn = nn.CrossEntropyLoss()
        loss_location = location_loss_fn(logits, target_indices)

        return delta_G, E_complex, euclidean_distances, gt_probs, pred_probs, loss_location

def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        avg_train_losses = checkpoint['avg_train_losses']
        avg_val_losses = checkpoint['avg_val_losses']
        processed_train_files = checkpoint['processed_train_files']

        print(f"Checkpoint loaded from epoch {epoch}")
        return epoch, train_losses, val_losses, avg_train_losses, avg_val_losses, processed_train_files
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}. Starting from scratch.")
        return 0, [], [], [], [], []

def save_checkpoint(epoch, model, optimizer, checkpoint_path, train_losses, val_losses, avg_train_losses, avg_val_losses, processed_train_files):
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'avg_train_losses': avg_train_losses,
        'avg_val_losses': avg_val_losses,
        'processed_train_files': processed_train_files
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def crop_tensor(tensor, center_idx, crop_size):
    """
    Crops a 4D tensor [C, X, Y, Z] around a spatial center index.
    """
    half = crop_size // 2
    cx, cy, cz = center_idx

    # Compute slice bounds
    x_start = max(0, cx - half)
    x_end = min(tensor.shape[1], cx + half + 1)
    y_start = max(0, cy - half)
    y_end = min(tensor.shape[2], cy + half + 1)
    z_start = max(0, cz - half)
    z_end = min(tensor.shape[3], cz + half + 1)

    # Extract the available region
    crop = tensor[:, x_start:x_end, y_start:y_end, z_start:z_end]

    # If crop is smaller than crop_size, pad it
    C = tensor.shape[0]
    if crop.shape[1:] != (crop_size, crop_size, crop_size):
        padded = torch.zeros(C, crop_size, crop_size, crop_size, 
                             dtype=tensor.dtype, device=tensor.device)
        
        px = (crop_size - crop.shape[1]) // 2
        py = (crop_size - crop.shape[2]) // 2
        pz = (crop_size - crop.shape[3]) // 2
        
        padded[:, px:px+crop.shape[1], 
                  py:py+crop.shape[2], 
                  pz:pz+crop.shape[3]] = crop
        return padded
    
    # 🛠️ MEMORY FIX: Force a clone so the massive parent tensor can be freed!
    return crop.clone()

def load_ensemble_data(pkl_path, device=None):
    """
    Loads individual conformer slice data from the new split format.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        complex_tensor = data.get("complex_sparse")
        protein_tensor = data.get("protein_sparse")
        peptide_tensor = data.get("peptide_sparse")
        
        if complex_tensor is None or protein_tensor is None or peptide_tensor is None:
            print(f"⚠️ Missing keys in {pkl_path}. Found: {list(data.keys())}")
            return None, None, None
        
        # Unpack layouts seamlessly
        if hasattr(complex_tensor, "is_sparse") and complex_tensor.is_sparse:
            complex_tensor = complex_tensor.to_dense()
        elif torch.is_tensor(complex_tensor) and complex_tensor.layout == torch.sparse_coo:
            complex_tensor = complex_tensor.to_dense()

        if hasattr(protein_tensor, "is_sparse") and protein_tensor.is_sparse:
            protein_tensor = protein_tensor.to_dense()
        elif torch.is_tensor(protein_tensor) and protein_tensor.layout == torch.sparse_coo:
            protein_tensor = protein_tensor.to_dense()

        if hasattr(peptide_tensor, "is_sparse") and peptide_tensor.is_sparse:
            peptide_tensor = peptide_tensor.to_dense()
        elif torch.is_tensor(peptide_tensor) and peptide_tensor.layout == torch.sparse_coo:
            peptide_tensor = peptide_tensor.to_dense()
        
        if device is not None:
            complex_tensor = complex_tensor.to(device)
            protein_tensor = protein_tensor.to(device)
            peptide_tensor = peptide_tensor.to(device)
        
        return complex_tensor, protein_tensor, peptide_tensor
        
    except Exception as e:
        print(f"❌ Error loading {pkl_path}: {e}")
        return None, None, None

def numerical_sort(value):
    """Sorts strings naturally (model_2 before model_10)"""
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else 0

def train_model(model, optimizer, num_epochs, checkpoint_path,
                train_pdb_ids, val_pdb_ids, DATA_FOLDERS, deltaG_dict, 
                examples_to_run=100, crop_size=31,alpha = 1): 

    start_epoch, train_losses, val_losses, avg_train_losses, avg_val_losses, processed_train_files = load_checkpoint(model, optimizer, checkpoint_path)
    dG_loss_fn = torch.nn.MSELoss() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0

        # ===================== TRAINING =====================
        for i, pdb_id in enumerate(train_pdb_ids):
            pdb_dir = None
            for folder in DATA_FOLDERS:
                potential_dir = os.path.join(folder, pdb_id.lower())
                if os.path.exists(potential_dir):
                    pdb_dir = potential_dir
                    break
            
            if pdb_dir is None:
                print(f"⚠️ Skipping {pdb_id}: no directory found")
                continue
            
            all_files = glob.glob(os.path.join(pdb_dir, "model_*_data.pkl"))
            conformer_files = sorted(all_files, key=numerical_sort)[:examples_to_run]
            
            if len(conformer_files) == 0:
                print(f"⚠️ Skipping {pdb_id}: no model_*_data.pkl files found")
                continue
            
            print(f"Loading {len(conformer_files)} conformer files for {pdb_id}...")
            complex_crops, protein_crops, peptide_crops = [], [], []
            
            # 🛠️ MEMORY FIX: Load onto CPU RAM (device=None) to protect GPU VRAM
            for pkl_file in conformer_files:
                c_tens, p_tens, l_tens = load_ensemble_data(pkl_file, device=None)
                if c_tens is None:
                    continue
                
                center_idx = tuple(s // 2 for s in c_tens.shape[-3:])
                
                complex_crops.append(crop_tensor(c_tens, center_idx, crop_size))
                protein_crops.append(crop_tensor(p_tens, center_idx, crop_size))
                peptide_crops.append(crop_tensor(l_tens, center_idx, crop_size))

                # 🛠️ MEMORY FIX: Instantly delete the massive dense background grids!
                del c_tens, p_tens, l_tens
            
            if len(complex_crops) == 0:
                print(f"⚠️ Skipping {pdb_id} due to loading errors.\n")
                continue
            
            # 🛠️ MEMORY FIX: Stack on CPU first, then transfer a single combined batch to GPU
            complex_batch = torch.stack(complex_crops).to(device)
            protein_batch = torch.stack(protein_crops).to(device)
            peptide_batch = torch.stack(peptide_crops).to(device)
            
            print(f"Batch shapes - Complex: {complex_batch.shape}, Protein: {protein_batch.shape}, Peptide: {peptide_batch.shape}")
            
            deltaG_pred, E_complex, euclidean_distances, gt_probs, pred_probs, loss_location = model(complex_batch, protein_batch, peptide_batch)
            
            print(f"Predicted ΔG for {pdb_id}: {deltaG_pred.item():.4f} kcal/mol")

            if pdb_id.upper() in deltaG_dict:
                deltaG_exp = torch.tensor(deltaG_dict[pdb_id.upper()], device=device, dtype=torch.float)
                assert deltaG_exp < 0, f"deltaG_exp must be negative, got {deltaG_exp}"
                loss_dG = dG_loss_fn(deltaG_pred, deltaG_exp)
            else:
                loss_dG = torch.tensor(0.0, device=device)

            loss = alpha * loss_location + loss_dG


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Delta G Bias:", model.deltaG_bias.item())

            epoch_train_loss += loss.item()
            train_losses.append(loss.item())
            processed_train_files.append(pdb_id)

            print(f"Epoch [{epoch+1}/{num_epochs}], File [{i+1}/{len(train_pdb_ids)}], "
                  f"Loss: {loss.item():.8f}, Euclidean Distance (mean): {euclidean_distances.mean():.2f}, "
                  f"GT Prob (Max): {gt_probs.max().item():.8e}, Pred Prob (Max): {pred_probs.max().item():.8e}")
            print(f"Loss components Location: {loss_location.item():.6f}, ΔG: {loss_dG.item():.6f}\n")

            del complex_batch, protein_batch, peptide_batch, loss, E_complex
            torch.cuda.empty_cache()

        avg_train_loss = epoch_train_loss / max(1, len(train_pdb_ids))
        avg_train_losses.append(avg_train_loss)

        # ===================== VALIDATION =====================
        model.eval()
        epoch_val_loss = 0
        val_count = 0

        with torch.no_grad():
            for i, pdb_id in enumerate(val_pdb_ids):
                pdb_dir = None
                for folder in DATA_FOLDERS:
                    potential_dir = os.path.join(folder, pdb_id.lower())
                    if os.path.exists(potential_dir):
                        pdb_dir = potential_dir
                        break
                
                if pdb_dir is None:
                    continue

                all_files = glob.glob(os.path.join(pdb_dir, "model_*_data.pkl"))
                conformer_files = sorted(all_files, key=numerical_sort)[:examples_to_run]
                
                if len(conformer_files) == 0:
                    continue

                print(f"Loading {len(conformer_files)} conformer files for {pdb_id} (Val)...")

                complex_crops, protein_crops, peptide_crops = [], [], []
                
                for pkl_file in conformer_files:
                    c_tens, p_tens, l_tens = load_ensemble_data(pkl_file, device=None)
                    if c_tens is None:
                        continue
                    
                    center_idx = tuple(s // 2 for s in c_tens.shape[-3:])
                    complex_crops.append(crop_tensor(c_tens, center_idx, crop_size))
                    protein_crops.append(crop_tensor(p_tens, center_idx, crop_size))
                    peptide_crops.append(crop_tensor(l_tens, center_idx, crop_size))

                    # 🛠️ MEMORY FIX: Instantly delete the massive dense background grids!
                    del c_tens, p_tens, l_tens

                if len(complex_crops) == 0:
                    continue

                complex_batch = torch.stack(complex_crops).to(device)
                protein_batch = torch.stack(protein_crops).to(device)
                peptide_batch = torch.stack(peptide_crops).to(device)

                deltaG_pred, E_complex, euclidean_distances, gt_probs, pred_probs, loss_location = model(complex_batch, protein_batch, peptide_batch)

                if pdb_id.upper() in deltaG_dict:
                    deltaG_exp = torch.tensor(deltaG_dict[pdb_id.upper()], device=device, dtype=torch.float)
                    loss_dG = dG_loss_fn(deltaG_pred, deltaG_exp)
                else:
                    loss_dG = torch.tensor(0.0, device=device)

                loss = alpha * loss_location + loss_dG


                epoch_val_loss += loss.item()
                val_losses.append(loss.item())
                val_count += 1

                print(f"Validation ΔG for {pdb_id}: {deltaG_pred.item():.4f} kcal/mol")
                print(f"Epoch [{epoch+1}/{num_epochs}], File [{i+1}/{len(val_pdb_ids)}], "
                      f"Val Loss: {loss.item():.8f}, Euclidean Distance (mean): {euclidean_distances.mean():.2f}")
                print(f"Loss components Location: {loss_location.item():.6f}, ΔG: {loss_dG.item():.6f}\n")

                del complex_batch, protein_batch, peptide_batch, loss, E_complex
                torch.cuda.empty_cache()

        avg_val_loss = epoch_val_loss / max(1, val_count)
        avg_val_losses.append(avg_val_loss)

        save_checkpoint(
            epoch + 1, model, optimizer, checkpoint_path,
            train_losses, val_losses, avg_train_losses, avg_val_losses, processed_train_files
        )

    return train_losses, val_losses, avg_train_losses, avg_val_losses
def test_model(model, test_pdb_ids, DATA_FOLDERS, deltaG_dict, examples_to_run=100, crop_size=31):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print("========== TEST RESULTS ==========")
    print("PDB_ID,Predicted_dG,Experimental_dG")

    with torch.no_grad():
        for pdb_id in test_pdb_ids:
            pdb_dir = None
            for folder in DATA_FOLDERS:
                potential_dir = os.path.join(folder, pdb_id.lower())
                if os.path.exists(potential_dir):
                    pdb_dir = potential_dir
                    break
            
            if pdb_dir is None:
                print(f"⚠️ Skipping {pdb_id}: directory not found")
                continue

            all_files = glob.glob(os.path.join(pdb_dir, "model_*_data.pkl"))
            conformer_files = sorted(all_files, key=numerical_sort)[:examples_to_run]
            
            if len(conformer_files) == 0:
                print(f"⚠️ Skipping {pdb_id}: no data files")
                continue

            complex_crops, protein_crops, peptide_crops = [], [], []
            
            # 🛠️ MEMORY FIX: Load onto CPU RAM (device=None) to protect GPU VRAM
            for pkl_file in conformer_files:
                c_tens, p_tens, l_tens = load_ensemble_data(pkl_file, device=None)
                if c_tens is None:
                    continue
                center_idx = tuple(s // 2 for s in c_tens.shape[-3:])
                complex_crops.append(crop_tensor(c_tens, center_idx, crop_size))
                protein_crops.append(crop_tensor(p_tens, center_idx, crop_size))
                peptide_crops.append(crop_tensor(l_tens, center_idx, crop_size))

                # 🛠️ MEMORY FIX: Instantly delete the massive dense background grids!
                del c_tens, p_tens, l_tens

            if len(complex_crops) == 0:
                continue

            # 🛠️ MEMORY FIX: Stack on CPU first, then transfer a single combined batch to GPU
            complex_batch = torch.stack(complex_crops).to(device)
            protein_batch = torch.stack(protein_crops).to(device)
            peptide_batch = torch.stack(peptide_crops).to(device)

            # 🛠️ SIGNATURE FIX: Unpack all 6 outputs matching your updated forward pass
            deltaG_pred, _, _, _, _, _ = model(complex_batch, protein_batch, peptide_batch)
            exp_dg = deltaG_dict.get(pdb_id.upper(), np.nan)

            results.append({
                "PDB_ID": pdb_id,
                "Predicted_dG": deltaG_pred.item(),
                "Experimental_dG": exp_dg
            })
            
            print(f"{pdb_id},{deltaG_pred.item():.4f},{exp_dg}")
            
            # Clear memory explicitly after processing this PDB folder
            del complex_batch, protein_batch, peptide_batch
            torch.cuda.empty_cache()

    print("=======================================\n")
    return results
def save_split(train_pdb_ids, val_pdb_ids, split_file=SPLIT_FILE):
    with open(split_file, 'w') as f:
        f.write("Training PDB IDs:\n")
        for pdb_id in train_pdb_ids:
            f.write(f"{pdb_id.lower()}\n")

        f.write("\nValidation PDB IDs:\n")
        for pdb_id in val_pdb_ids:
            f.write(f"{pdb_id.lower()}\n")
    print(f"Split saved to {split_file}")

def load_split(split_file=SPLIT_FILE):
    if not os.path.exists(split_file):
        return None, None

    train_pdb_ids, val_pdb_ids = [], []
    with open(split_file, 'r') as f:
        lines = f.readlines()
        current_section = None

        for line in lines:
            line = line.strip()
            if "Training PDB IDs" in line:
                current_section = "train"
            elif "Validation PDB IDs" in line:
                current_section = "val"
            elif line:
                if current_section == "train":
                    train_pdb_ids.append(line.lower())
                elif current_section == "val":
                    val_pdb_ids.append(line.lower())

    print(f"Split loaded from {split_file}")
    return train_pdb_ids, val_pdb_ids

def main():
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="Train a 3D convolutional docking model with ensemble energies.")
    parser.add_argument('--data_folders', nargs='+', required=True, help='Paths to the data folders (space-separated).')
    parser.add_argument('--cluster_file', type=str, required=True, help='Path to CSV with Cluster IDs.')
    parser.add_argument('--checkpoint_path', type=str, default='Adamw_checkpoint.pkl')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--examples_to_run', type=int, default=100, help='Number of conformers to use per ensemble')
    parser.add_argument('--crop_size', type=int, default=31, help='Size of the cropped region')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weighting factor for loss_location vs loss_dG')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_pdb_ids, val_pdb_ids = None, None
    try:
        train_pdb_ids, val_pdb_ids = load_split()
    except Exception as e:
        print(f"load_split() failed, building from cluster file: {e}")

    if train_pdb_ids is None or val_pdb_ids is None:
        cluster_data = pd.read_csv(args.cluster_file)
        if 'PDB Code' in cluster_data.columns: cluster_data['PDB_ID'] = cluster_data['PDB Code']
        cluster_data['PDB_ID'] = cluster_data['PDB_ID'].str.upper().str.strip()

        unique_clusters = cluster_data['Cluster ID'].unique()
        np.random.shuffle(unique_clusters)

        num_train = int(0.95 * len(unique_clusters)) if len(unique_clusters) > 1 else 1
        train_data = cluster_data[cluster_data['Cluster ID'].isin(unique_clusters[:num_train])]
        val_data = cluster_data[cluster_data['Cluster ID'].isin(unique_clusters[num_train:])]
        
        train_pdb_ids, val_pdb_ids = list(train_data['PDB_ID'].unique()), list(val_data['PDB_ID'].unique())
        save_split(train_pdb_ids, val_pdb_ids)
    print(f"Training PDBs: {len(train_pdb_ids)} | Validation/Test PDBs: {len(val_pdb_ids)}")

    DATA_FOLDERS = args.data_folders 
    
    df = pd.read_csv("experimental_data.csv")
    df['PDB Code'] = df['PDB Code'].str.upper().str.strip()
    deltaG_dict = dict(zip(df['PDB Code'], df['DeltaG (kcal/mol)'].astype(float)))

    model = Docking(dockingFFT=None).to(device)

    # Single optimizer with two parameter groups:
    # main model uses args.learning_rate, bias uses a 10x slower rate
    optimizer = torch.optim.AdamW([
        {'params': [p for name, p in model.named_parameters() if name != "deltaG_bias"],
         'lr': args.learning_rate},
        {'params': [model.deltaG_bias],
         'lr': args.learning_rate * 0.1}  # 10x slower — bias should drift, not jump
    ])
    # Run training
    train_losses, val_losses, avg_train_losses, avg_val_losses = train_model(
        model=model, optimizer=optimizer,
        num_epochs=args.num_epochs,
        checkpoint_path=args.checkpoint_path, train_pdb_ids=train_pdb_ids,
        val_pdb_ids=val_pdb_ids, DATA_FOLDERS=DATA_FOLDERS, deltaG_dict=deltaG_dict,
        examples_to_run=args.examples_to_run, crop_size=args.crop_size, alpha=args.alpha
    )

    # Save training plots
    output_prefix = f"dock_model_{args.num_epochs}ep_{args.learning_rate}lr"
    
    losses_df = pd.DataFrame({
        'Epoch': range(1, len(avg_train_losses) + 1),
        'Avg_Train_Loss': avg_train_losses,
        'Avg_Val_Loss': avg_val_losses
    })
    losses_df.to_csv(f"training_losses_{output_prefix}.csv", index=False)
    print(f"✅ Training losses saved to training_losses_{output_prefix}.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_losses, label='Average Train Loss')
    plt.plot(avg_val_losses, label='Average Val Loss')
    plt.legend()
    plt.savefig(f"avg_loss_{output_prefix}.png")
    plt.close()

    # Run final test
    final_test_results = test_model(
        model=model, 
        test_pdb_ids=val_pdb_ids, 
        DATA_FOLDERS=DATA_FOLDERS, 
        deltaG_dict=deltaG_dict,
        examples_to_run=args.examples_to_run,
        crop_size=args.crop_size
    )

    if final_test_results:
        results_df = pd.DataFrame(final_test_results)
        csv_filename = f"final_test_predictions_{output_prefix}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"✅ Final predictions successfully saved to {csv_filename}")

    print("🎉 Pipeline completed.")

if __name__ == "__main__":
    main()