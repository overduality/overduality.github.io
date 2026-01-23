import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CBAM ATTENTION MODULE
# ============================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ============================================
# EFFICIENTNET-B3 + CBAM MODEL
# ============================================
class EfficientNetB3_CBAM(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b3', 
                                          pretrained=pretrained, 
                                          num_classes=0,
                                          global_pool='')
        
        self.num_features = self.backbone.num_features
        self.cbam = CBAM(self.num_features, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.num_features, 1)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        attended = self.cbam(features)
        pooled = self.global_pool(attended)
        pooled = pooled.view(pooled.size(0), -1)
        
        out = self.dropout(pooled)
        age = self.fc(out).squeeze(-1)
        
        if return_features:
            return age, attended
        return age

# ============================================
# DATASET CLASS
# ============================================
class DentalAgeDataset(Dataset):
    def __init__(self, df, transform=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
        
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = row['filepath']
        img = Image.open(img_path).convert('L')
        
        if self.augment:
            img = self.augment_transform(img)
        
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        age = torch.tensor(row['age'], dtype=torch.float32)
        patient_id = row['patient_id']
        
        return img, age, patient_id

# ============================================
# TRAINER CLASS
# ============================================
class DentalAgeTrainer:
    def __init__(self, device='mps', model_save_dir='trained_models'):
        self.device = device
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def train_one_epoch(self, model, loader, criterion, optimizer, scheduler=None):
        model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(loader, desc='Training', leave=False)
        for imgs, ages, _ in pbar:
            imgs, ages = imgs.to(self.device), ages.to(self.device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, ages)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(preds.detach().cpu().numpy())
            targets.extend(ages.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if scheduler is not None:
            scheduler.step()
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        
        return total_loss / len(loader), mae, rmse
    
    def validate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        predictions = []
        targets = []
        patient_ids = []
        
        with torch.no_grad():
            for imgs, ages, pids in tqdm(loader, desc='Validating', leave=False):
                imgs, ages = imgs.to(self.device), ages.to(self.device)
                
                preds = model(imgs)
                loss = criterion(preds, ages)
                
                total_loss += loss.item()
                predictions.extend(preds.cpu().numpy())
                targets.extend(ages.cpu().numpy())
                patient_ids.extend(pids)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        
        results_df = pd.DataFrame({
            'patient_id': patient_ids,
            'true_age': targets,
            'pred_age': predictions,
            'error': predictions - targets
        })
        
        return total_loss / len(loader), mae, rmse, results_df
    
    def train_fold(self, train_df, val_df, fold_num, n_epochs=100, 
                   lr=1e-4, patience=15, batch_size=16):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}")
        print(f"{'='*60}")
        print(f"Train: {len(train_df)} images, {train_df['patient_id'].nunique()} patients")
        print(f"Val:   {len(val_df)} images, {val_df['patient_id'].nunique()} patients")
        
        train_patients = set(train_df['patient_id'].unique())
        val_patients = set(val_df['patient_id'].unique())
        overlap = train_patients & val_patients
        assert len(overlap) == 0, f"Patient leakage detected: {overlap}"
        print("✅ No patient leakage verified")
        
        train_dataset = DentalAgeDataset(train_df, self.train_transform, augment=True)
        val_dataset = DentalAgeDataset(val_df, self.val_transform, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0, pin_memory=True)
        
        model = EfficientNetB3_CBAM(pretrained=True, dropout=0.4).to(self.device)
        
        criterion = nn.SmoothL1Loss(beta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2, eta_min=1e-6
        )
        
        best_val_mae = float('inf')
        patience_counter = 0
        history = []
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            train_loss, train_mae, train_rmse = self.train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler
            )
            
            val_loss, val_mae, val_rmse, val_results = self.validate(
                model, val_loader, criterion
            )
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                
                save_path = self.model_save_dir / f'fold_{fold_num}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_mae': float(val_mae),
                    'val_rmse': float(val_rmse),
                }, save_path)
                print(f"✅ Saved best model (MAE: {val_mae:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        checkpoint = torch.load(self.model_save_dir / f'fold_{fold_num}_best.pth', 
                               weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        _, final_mae, final_rmse, final_results = self.validate(
            model, val_loader, criterion
        )
        
        return {
            'fold': fold_num,
            'best_val_mae': best_val_mae,
            'best_val_rmse': checkpoint['val_rmse'],
            'final_mae': final_mae,
            'final_rmse': final_rmse,
            'best_epoch': checkpoint['epoch'],
            'history': history,
            'val_results': final_results
        }
    
    def cross_validate(self, train_csv, n_splits=5, n_epochs=100, 
                      lr=1e-4, patience=15, batch_size=16):
        print("="*80)
        print("STRATIFIED GROUP K-FOLD CROSS-VALIDATION")
        print("="*80)
        
        df = pd.read_csv(train_csv)
        df['age_bin'] = pd.qcut(df['age'], q=4, labels=False, duplicates='drop')
        
        X = df.index.values
        y = df['age_bin'].values
        groups = df['patient_id'].values
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_num, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            result = self.train_fold(
                train_df, val_df, fold_num,
                n_epochs=n_epochs, lr=lr, patience=patience, batch_size=batch_size
            )
            
            fold_results.append(result)
        
        self.summarize_cv_results(fold_results)
        
        return fold_results
    
    def summarize_cv_results(self, fold_results):
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        maes = [r['best_val_mae'] for r in fold_results]
        rmses = [r['best_val_rmse'] for r in fold_results]
        
        print(f"\n📊 MAE per fold:")
        for i, mae in enumerate(maes, 1):
            print(f"  Fold {i}: {mae:.4f} years")
        print(f"\n  Mean MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f} years")
        
        print(f"\n📊 RMSE per fold:")
        for i, rmse in enumerate(rmses, 1):
            print(f"  Fold {i}: {rmse:.4f} years")
        print(f"\n  Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f} years")
        
        summary = pd.DataFrame({
            'fold': [r['fold'] for r in fold_results],
            'best_val_mae': maes,
            'best_val_rmse': rmses,
            'best_epoch': [r['best_epoch'] for r in fold_results]
        })
        summary.to_csv(self.model_save_dir / 'cv_summary.csv', index=False)
        
        self.plot_training_curves(fold_results)
        self.plot_bland_altman_cv(fold_results)
    
    def plot_training_curves(self, fold_results):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, result in enumerate(fold_results):
            ax = axes[idx // 3, idx % 3]
            history = pd.DataFrame(result['history'])
            
            ax.plot(history['epoch'], history['train_mae'], 
                   label='Train MAE', color='blue', alpha=0.7)
            ax.plot(history['epoch'], history['val_mae'], 
                   label='Val MAE', color='red', alpha=0.7)
            ax.axvline(x=result['best_epoch'], color='green', 
                      linestyle='--', alpha=0.5, label='Best')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE (years)')
            ax.set_title(f'Fold {result["fold"]} (Best MAE: {result["best_val_mae"]:.3f})')
            ax.legend()
            ax.grid(alpha=0.3)
        
        if len(fold_results) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {self.model_save_dir / 'training_curves.png'}")
    
    def plot_bland_altman_cv(self, fold_results):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, result in enumerate(fold_results):
            ax = axes[idx // 3, idx % 3]
            df = result['val_results']
            
            mean_age = (df['true_age'] + df['pred_age']) / 2
            diff_age = df['pred_age'] - df['true_age']
            
            mean_diff = diff_age.mean()
            std_diff = diff_age.std()
            
            ax.scatter(mean_age, diff_age, alpha=0.5, s=30)
            ax.axhline(y=mean_diff, color='red', linestyle='--', 
                      label=f'Mean: {mean_diff:.3f}')
            ax.axhline(y=mean_diff + 1.96*std_diff, color='gray', 
                      linestyle='--', alpha=0.7)
            ax.axhline(y=mean_diff - 1.96*std_diff, color='gray', 
                      linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            ax.set_xlabel('Mean Age (years)')
            ax.set_ylabel('Difference (Predicted - True)')
            ax.set_title(f'Fold {result["fold"]} Bland-Altman')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        if len(fold_results) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'bland_altman_cv.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {self.model_save_dir / 'bland_altman_cv.png'}")


if __name__ == "__main__":
    TRAIN_CSV = 'dataset_splits/train_set.csv'
    MODEL_SAVE_DIR = 'trained_models'
    
    N_SPLITS = 5
    N_EPOCHS = 150
    LEARNING_RATE = 1e-4
    PATIENCE = 15
    BATCH_SIZE = 16
    
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    trainer = DentalAgeTrainer(device=DEVICE, model_save_dir=MODEL_SAVE_DIR)
    
    cv_results = trainer.cross_validate(
        train_csv=TRAIN_CSV,
        n_splits=N_SPLITS,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        batch_size=BATCH_SIZE
    )
    
    print("\n✅ Training complete! Run evaluate_model.py to test on locked test set.")