import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import model architecture from training script
from train_obb import EfficientNetB3_CBAM, DentalAgeDataset

# ============================================
# TEST SET EVALUATOR
# ============================================
class TestSetEvaluator:
    def __init__(self, device='mps', model_save_dir='trained_models'):
        self.device = device
        self.model_save_dir = Path(model_save_dir)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def mc_dropout_predict(self, model, img_tensor, n_samples=30):
        """Predict with MC Dropout for uncertainty estimation"""
        model.eval()
        
        # Enable dropout during inference
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(img_tensor).item()
                predictions.append(pred)
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return mean_pred, std_pred
    
    def evaluate_ensemble(self, test_csv, n_folds=5, use_mc_dropout=True, n_mc_samples=30):
        """Evaluate test set using ensemble of fold models"""
        print("="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        
        df = pd.read_csv(test_csv)
        print(f"\nTest set: {len(df)} images, {df['patient_id'].nunique()} patients")
        
        test_dataset = DentalAgeDataset(df, self.transform, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Load all fold models
        models = []
        for fold_num in range(1, n_folds + 1):
            model_path = self.model_save_dir / f'fold_{fold_num}_best.pth'
            
            if not model_path.exists():
                print(f"⚠️  Warning: {model_path} not found, skipping...")
                continue
            
            model = EfficientNetB3_CBAM(pretrained=False, dropout=0.4).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        
        print(f"✅ Loaded {len(models)} fold models")
        
        # Predictions
        results = []
        
        for imgs, ages, pids in tqdm(test_loader, desc='Evaluating'):
            imgs = imgs.to(self.device)
            true_age = ages.item()
            patient_id = pids[0]
            
            fold_preds = []
            fold_uncertainties = []
            
            for model in models:
                if use_mc_dropout:
                    mean_pred, std_pred = self.mc_dropout_predict(model, imgs, n_mc_samples)
                    fold_preds.append(mean_pred)
                    fold_uncertainties.append(std_pred)
                else:
                    with torch.no_grad():
                        pred = model(imgs).item()
                    fold_preds.append(pred)
            
            ensemble_mean = np.mean(fold_preds)
            ensemble_std = np.std(fold_preds)
            
            if use_mc_dropout:
                aleatoric = np.mean(fold_uncertainties)
                epistemic = ensemble_std
                total_uncertainty = np.sqrt(aleatoric**2 + epistemic**2)
            else:
                total_uncertainty = ensemble_std
            
            results.append({
                'patient_id': patient_id,
                'true_age': true_age,
                'pred_age': ensemble_mean,
                'uncertainty': total_uncertainty,
                'error': ensemble_mean - true_age,
                'abs_error': abs(ensemble_mean - true_age)
            })
        
        results_df = pd.DataFrame(results)
        
        mae = results_df['abs_error'].mean()
        rmse = np.sqrt((results_df['error']**2).mean())
        
        print("\n" + "="*80)
        print("FINAL TEST SET RESULTS")
        print("="*80)
        print(f"MAE:  {mae:.4f} years")
        print(f"RMSE: {rmse:.4f} years")
        print(f"Mean Uncertainty: {results_df['uncertainty'].mean():.4f} years")
        
        results_df.to_csv(self.model_save_dir / 'test_set_results.csv', index=False)
        print(f"\n💾 Saved: {self.model_save_dir / 'test_set_results.csv'}")
        
        self.plot_test_results(results_df)
        
        return results_df
    
    def plot_test_results(self, results_df):
        """Generate comprehensive test result visualizations"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Scatter plot with uncertainty
        ax1 = plt.subplot(2, 3, 1)
        ax1.errorbar(results_df['true_age'], results_df['pred_age'], 
                     yerr=results_df['uncertainty'], fmt='o', alpha=0.6,
                     elinewidth=1, capsize=3)
        ax1.plot([3, 16], [3, 16], 'r--', alpha=0.5, label='Perfect prediction')
        ax1.set_xlabel('True Age (years)')
        ax1.set_ylabel('Predicted Age (years)')
        ax1.set_title('Predictions with Uncertainty')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Bland-Altman plot
        ax2 = plt.subplot(2, 3, 2)
        mean_age = (results_df['true_age'] + results_df['pred_age']) / 2
        diff_age = results_df['error']
        mean_diff = diff_age.mean()
        std_diff = diff_age.std()
        
        ax2.scatter(mean_age, diff_age, alpha=0.6)
        ax2.axhline(y=mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.3f}')
        ax2.axhline(y=mean_diff + 1.96*std_diff, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=mean_diff - 1.96*std_diff, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Mean Age (years)')
        ax2.set_ylabel('Difference (Predicted - True)')
        ax2.set_title('Bland-Altman Plot')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Error distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(results_df['error'], bins=20, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Error (years)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Error Distribution (MAE: {results_df["abs_error"].mean():.3f})')
        ax3.grid(alpha=0.3)
        
        # 4. Uncertainty vs Error
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(results_df['uncertainty'], results_df['abs_error'], alpha=0.6)
        ax4.set_xlabel('Predicted Uncertainty (years)')
        ax4.set_ylabel('Absolute Error (years)')
        ax4.set_title('Uncertainty Calibration')
        ax4.grid(alpha=0.3)
        
        # 5. Error by age group
        ax5 = plt.subplot(2, 3, 5)
        age_bins = pd.cut(results_df['true_age'], bins=[3, 6, 9, 12, 16])
        results_df['age_group'] = age_bins
        age_group_errors = results_df.groupby('age_group')['abs_error'].agg(['mean', 'std'])
        age_group_errors.plot(kind='bar', y='mean', yerr='std', ax=ax5, capsize=4, alpha=0.7)
        ax5.set_xlabel('Age Group')
        ax5.set_ylabel('Mean Absolute Error (years)')
        ax5.set_title('Error by Age Group')
        ax5.grid(alpha=0.3, axis='y')
        ax5.legend().remove()
        
        # 6. Q-Q plot for normality check
        ax6 = plt.subplot(2, 3, 6)
        stats.probplot(results_df['error'], dist="norm", plot=ax6)
        ax6.set_title('Q-Q Plot (Normality Check)')
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'test_set_evaluation.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {self.model_save_dir / 'test_set_evaluation.png'}")


# ============================================
# GRAD-CAM VISUALIZER
# ============================================
class GradCAMVisualizer:
    def __init__(self, model, device='mps'):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks on CBAM output
        self.model.cbam.spatial_attention.register_forward_hook(self.save_activation)
        self.model.cbam.spatial_attention.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, img_tensor, true_age):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        img_tensor = img_tensor.to(self.device)
        
        pred_age = self.model(img_tensor)
        
        self.model.zero_grad()
        pred_age.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        
        return cam, pred_age.item()
    
    def visualize_samples(self, test_csv, n_samples=12, output_path='gradcam_samples.png'):
        """Visualize Grad-CAM for random test samples"""
        df = pd.read_csv(test_csv)
        samples = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            img = Image.open(row['filepath']).convert('L').convert('RGB')
            img_np = np.array(img)
            
            img_tensor = transform(img).unsqueeze(0)
            
            cam, pred_age = self.generate_cam(img_tensor, row['age'])
            
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224)))
            
            heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
            overlay = (img_np / 255.0) * 0.5 + heatmap * 0.5
            
            axes[idx].imshow(overlay)
            axes[idx].set_title(f"True: {row['age']:.1f}y | Pred: {pred_age:.1f}y\n"
                               f"Error: {pred_age - row['age']:.2f}y", fontsize=9)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    TEST_CSV = 'dataset_splits/test_set.csv'
    MODEL_SAVE_DIR = 'trained_models'
    N_FOLDS = 5
    
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    print("\n" + "="*80)
    print("⚠️  LOCKED TEST SET EVALUATION")
    print("="*80)
    print("This test set should ONLY be evaluated ONCE!")
    print("If test performance is much worse than CV, you overfitted.")
    print("="*80)
    
    response = input("\nProceed with test set evaluation? (yes/no): ")
    
    if response.lower() == 'yes':
        evaluator = TestSetEvaluator(device=DEVICE, model_save_dir=MODEL_SAVE_DIR)
        
        test_results = evaluator.evaluate_ensemble(
            test_csv=TEST_CSV,
            n_folds=N_FOLDS,
            use_mc_dropout=True,
            n_mc_samples=30
        )
        
        print("\n" + "="*80)
        print("GRAD-CAM VISUALIZATION")
        print("="*80)
        
        model_path = Path(MODEL_SAVE_DIR) / 'fold_1_best.pth'
        model = EfficientNetB3_CBAM(pretrained=False, dropout=0.4).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        visualizer = GradCAMVisualizer(model, device=DEVICE)
        visualizer.visualize_samples(
            test_csv=TEST_CSV,
            n_samples=12,
            output_path=Path(MODEL_SAVE_DIR) / 'gradcam_test_samples.png'
        )
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED!")
        print("="*80)
    else:
        print("\n✅ Test evaluation cancelled. Review CV results first!")