import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import os
import json
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm
import logging
from datetime import datetime

from elder_assistance_cnn import ElderAssistanceCNN, MultiTaskLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElderAssistanceDataset(Dataset):
    """
    Multi-task dataset for Elder Assistance CNN
    
    Expected data structure:
    data_root/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── annotations/
    │   ├── person_reid.json
    │   ├── presence.json
    │   ├── gestures.json
    │   ├── fall_detection.json
    │   ├── objects.json
    │   ├── segmentation/
    │   ├── hazards.json
    │   ├── elevator.json
    │   └── compartment.json
    """
    
    def __init__(self, data_root: str, split: str = 'train', 
                 transform: Optional[transforms.Compose] = None,
                 tasks: List[str] = None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.tasks = tasks or ['all']
        
        self.images_path = os.path.join(data_root, 'dataset', split)
        print("[+] self.images_path:", self.images_path)
        self.annotations_path = os.path.join(data_root, 'annotations')
        
        # Load image paths
        self.image_paths = [
            os.path.join(self.images_path, f)
            for f in os.listdir(self.images_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print("[+] self.image_paths:", self.image_paths)
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Filter images that have required annotations
        self.valid_images = self._filter_valid_images()
        
        logger.info(f"Loaded {len(self.valid_images)} valid images for {split} split")
    
    def _load_annotations(self) -> Dict:
        """Load all annotation files"""
        annotations = {}
        
        annotation_files = {
            'person_reid': 'person_reid.json',
            'presence': 'presence.json', 
            'gesture': 'gestures.json',
            'fall': 'fall_detection.json',
            'object_detection': 'objects.json',
            'hazard': 'hazards.json',
            'elevator': 'elevator.json',
            'compartment': 'compartment.json'
        }
        
        for task, filename in annotation_files.items():
            filepath = os.path.join(self.annotations_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    annotations[task] = json.load(f)
                logger.info(f"Loaded {task} annotations: {len(annotations[task])} entries")
            else:
                logger.warning(f"Annotation file not found: {filepath}")
                annotations[task] = {}
        
        # Load segmentation masks
        seg_path = os.path.join(self.annotations_path, 'segmentation')
        if os.path.exists(seg_path):
            annotations['segmentation'] = {
                f.replace('.png', ''): os.path.join(seg_path, f)
                for f in os.listdir(seg_path) if f.endswith('.png')
            }
        
        return annotations
    
    def _filter_valid_images(self) -> List[str]:
        """Filter images that have required annotations"""
        valid_images = []
        
        for img_path in self.image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Check if image has annotations for required tasks
            has_annotations = True
            if 'all' not in self.tasks:
                for task in self.tasks:
                    if task not in self.annotations or img_name not in self.annotations[task]:
                        has_annotations = False
                        break
            
            if has_annotations:
                valid_images.append(img_path)
        
        return valid_images
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_path = self.valid_images[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Prepare targets
        targets = {}
        
        # Person Re-ID
        if 'person_reid' in self.annotations and img_name in self.annotations['person_reid']:
            targets['person_reid'] = torch.tensor(
                self.annotations['person_reid'][img_name]['person_id'], dtype=torch.long
            )
        
        # Presence detection
        if 'presence' in self.annotations and img_name in self.annotations['presence']:
            targets['presence'] = torch.tensor(
                self.annotations['presence'][img_name]['present'], dtype=torch.long
            )
        
        # Gesture recognition
        if 'gesture' in self.annotations and img_name in self.annotations['gesture']:
            targets['gesture'] = torch.tensor(
                self.annotations['gesture'][img_name]['gesture_class'], dtype=torch.long
            )
        
        # Fall detection
        if 'fall' in self.annotations and img_name in self.annotations['fall']:
            targets['fall'] = torch.tensor(
                self.annotations['fall'][img_name]['fall_state'], dtype=torch.long
            )
        
        # Hazard detection (multi-label)
        if 'hazard' in self.annotations and img_name in self.annotations['hazard']:
            hazard_labels = self.annotations['hazard'][img_name]['hazards']
            targets['hazard'] = torch.tensor(hazard_labels, dtype=torch.float)
        
        # Elevator detection
        if 'elevator' in self.annotations and img_name in self.annotations['elevator']:
            targets['elevator'] = torch.tensor(
                self.annotations['elevator'][img_name]['elevator_state'], dtype=torch.long
            )
        
        # Compartment state (multi-label)
        if 'compartment' in self.annotations and img_name in self.annotations['compartment']:
            comp_labels = self.annotations['compartment'][img_name]['states']
            targets['compartment'] = torch.tensor(comp_labels, dtype=torch.float)
        
        # Segmentation
        if 'segmentation' in self.annotations and img_name in self.annotations['segmentation']:
            mask_path = self.annotations['segmentation'][img_name]
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize((224, 224))(mask)
            targets['segmentation'] = torch.tensor(np.array(mask), dtype=torch.long)
        
        # Object detection (simplified - you'll need to implement proper format)
        if 'object_detection' in self.annotations and img_name in self.annotations['object_detection']:
            # This is a placeholder - implement proper object detection target format
            targets['object_detection'] = torch.zeros(14, 14, 25)  # Grid format
        
        return image, targets


def get_transforms(split: str = 'train'):
    """Get data transforms for training/validation"""
    
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


class Trainer:
    """Training pipeline for Elder Assistance CNN"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = MultiTaskLoss()
        self.criterion.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create save directory
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        task_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Move targets to device
            targets_device = {}
            for task, target in targets.items():
                if isinstance(target, torch.Tensor):
                    targets_device[task] = target.to(self.device)
                else:
                    targets_device[task] = target
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(images)
                losses = self.criterion(outputs, targets_device)
                loss = losses['total']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            
            for task, task_loss in losses.items():
                if task != 'total':
                    if task not in task_losses:
                        task_losses[task] = 0
                    task_losses[task] += task_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        # Log task losses
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
        for task, task_loss in task_losses.items():
            avg_task_loss = task_loss / len(self.train_loader)
            logger.info(f"  {task}: {avg_task_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        task_losses = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
            for images, targets in pbar:
                images = images.to(self.device)
                
                # Move targets to device
                targets_device = {}
                for task, target in targets.items():
                    if isinstance(target, torch.Tensor):
                        targets_device[task] = target.to(self.device)
                    else:
                        targets_device[task] = target
                
                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets_device)
                    loss = losses['total']
                
                total_loss += loss.item()
                
                for task, task_loss in losses.items():
                    if task != 'total':
                        if task not in task_losses:
                            task_losses[task] = 0
                        task_losses[task] += task_loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Log validation results
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}")
        for task, task_loss in task_losses.items():
            avg_task_loss = task_loss / len(self.val_loader)
            logger.info(f"  {task}: {avg_task_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
            # Log epoch summary
            logger.info(f"Epoch {epoch}/{self.config['epochs']} - "
                       f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        logger.info("Training completed!")


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_root': '/Users/ryanh./CQU-models/Group 8/',  # Update this path
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'save_dir': 'checkpoints',
        'num_workers': 4
    }
    
    # Model configuration
    model_config = {
        'num_person_ids': 1000,
        'input_size': (224, 224),
        'num_classes_gesture': 4,
        'num_objects': 20,
        'num_segments': 10
    }
    
    # Create datasets
    train_dataset = ElderAssistanceDataset(
        data_root=config['data_root'],
        split='train',
        transform=get_transforms('train')
    )
    
    val_dataset = ElderAssistanceDataset(
        data_root=config['data_root'],
        split='val',
        transform=get_transforms('val')
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = ElderAssistanceCNN(**model_config)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()