import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import numpy as np

class ElderAssistanceCNN(nn.Module):
    """
    Multi-task CNN for Elder Assistance Robot
    
    Tasks:
    1. Person Re-ID / Tracking
    2. User Presence Verification
    3. Gesture Recognition
    4. Fall Detection
    5. Object Detection
    6. Semantic Segmentation
    7. Hazard Detection
    8. Elevator/Escalator Recognition
    9. Compartment State Monitoring
    """
    
    def __init__(self, num_person_ids=1000, input_size=(224, 224), num_classes_gesture=4, 
                 num_objects=20, num_segments=10):
        super(ElderAssistanceCNN, self).__init__()
        
        self.input_size = input_size
        self.num_person_ids = num_person_ids
        self.num_classes_gesture = num_classes_gesture  # Stop, Come, None, Other
        self.num_objects = num_objects
        self.num_segments = num_segments
        
        # Shared backbone - EfficientNet-B4 for efficiency
        self.backbone = models.efficientnet_b4(pretrained=True)
        self.backbone_features = self.backbone.features
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *input_size)
            backbone_out = self.backbone_features(dummy_input)
            self.feature_dim = backbone_out.shape[1]
            self.feature_spatial = (backbone_out.shape[2], backbone_out.shape[3])
        
        # Feature Pyramid Network for multi-scale features
        self.fpn = self._build_fpn()
        
        # Task-specific heads
        self._build_task_heads()
        
        # Attention mechanisms
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        
    def _build_fpn(self):
        """Feature Pyramid Network for multi-scale features"""
        fpn_layers = nn.ModuleDict()
        
        # Get intermediate features from EfficientNet
        fpn_layers['conv1'] = nn.Conv2d(self.feature_dim, 256, 1)
        fpn_layers['conv2'] = nn.Conv2d(256, 256, 3, padding=1)
        fpn_layers['conv3'] = nn.Conv2d(256, 256, 3, padding=1)
        
        return fpn_layers
    
    def _build_task_heads(self):
        """Build task-specific heads"""
        
        # 1. Person Re-ID Head
        self.person_reid_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # Feature embedding for Re-ID
            nn.BatchNorm1d(256)
        )
        
        # Person ID classifier (for training)
        self.person_classifier = nn.Linear(256, self.num_person_ids)
        
        # 2. User Presence Verification Head
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Present/Not Present
        )
        
        # 3. Gesture Recognition Head
        self.gesture_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes_gesture)
        )
        
        # 4. Fall Detection Head
        self.fall_detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Normal, Sitting, Fallen
        )
        
        # 5. Object Detection Head (simplified YOLO-style)
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_objects + 5, 1)  # classes + box coords + confidence
        )
        
        # 6. Semantic Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_segments, 1),
            nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False)
        )
        
        # 7. Hazard Detection Head
        self.hazard_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # stairs, ramp, curb, wet, narrow, safe
        )
        
        # 8. Elevator/Escalator Recognition Head
        self.elevator_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # elevator_open, elevator_closed, escalator, neither
        )
        
        # 9. Compartment State Head
        self.compartment_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # door_open, door_closed, hands_detected
        )
    
    def forward(self, x, task='all'):
        """Forward pass with task selection"""
        batch_size = x.shape[0]
        
        # Extract features using backbone
        features = self.backbone_features(x)
        
        # Feature pyramid processing
        fpn_features = self.fpn['conv1'](features)
        fpn_features = F.relu(fpn_features)
        fpn_features = self.fpn['conv2'](fpn_features)
        fpn_features = F.relu(fpn_features)
        
        outputs = {}
        
        if task == 'all' or task == 'person_reid':
            reid_features = self.person_reid_head(features)
            reid_logits = self.person_classifier(reid_features)
            outputs['person_reid'] = {
                'features': F.normalize(reid_features, p=2, dim=1),
                'logits': reid_logits
            }
        
        if task == 'all' or task == 'presence':
            outputs['presence'] = self.presence_head(features)
        
        if task == 'all' or task == 'gesture':
            outputs['gesture'] = self.gesture_head(features)
        
        if task == 'all' or task == 'fall':
            outputs['fall'] = self.fall_detection_head(features)
        
        if task == 'all' or task == 'object_detection':
            outputs['object_detection'] = self.object_detection_head(fpn_features)
        
        if task == 'all' or task == 'segmentation':
            outputs['segmentation'] = self.segmentation_head(fpn_features)
        
        if task == 'all' or task == 'hazard':
            outputs['hazard'] = self.hazard_head(features)
        
        if task == 'all' or task == 'elevator':
            outputs['elevator'] = self.elevator_head(features)
        
        if task == 'all' or task == 'compartment':
            outputs['compartment'] = self.compartment_head(features)
        
        return outputs


class PersonReIDLoss(nn.Module):
    """Triplet loss for person re-identification"""
    def __init__(self, margin=0.3):
        super(PersonReIDLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        
    def forward(self, features, labels):
        # Simple triplet sampling (in practice, use hard negative mining)
        batch_size = features.shape[0]
        
        # Create triplets
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Find positive (same person)
            pos_indices = (labels == anchor_label).nonzero().squeeze()
            if len(pos_indices.shape) == 0:
                pos_indices = pos_indices.unsqueeze(0)
            
            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
            
            # Find negative (different person)
            neg_indices = (labels != anchor_label).nonzero().squeeze()
            if len(neg_indices.shape) == 0:
                neg_indices = neg_indices.unsqueeze(0)
            
            if len(neg_indices) > 0:
                neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
            else:
                neg_idx = torch.randint(batch_size, (1,))
            
            anchors.append(features[i])
            positives.append(features[pos_idx])
            negatives.append(features[neg_idx])
        
        anchor_tensor = torch.stack(anchors)
        positive_tensor = torch.stack(positives)
        negative_tensor = torch.stack(negatives)
        
        return self.triplet_loss(anchor_tensor, positive_tensor, negative_tensor)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with adaptive weighting"""
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.reid_loss = PersonReIDLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        # Task weights (learnable)
        self.task_weights = nn.Parameter(torch.ones(9))
        
    def forward(self, outputs, targets):
        losses = {}
        total_loss = 0
        
        # Person Re-ID loss
        if 'person_reid' in outputs and 'person_reid' in targets:
            reid_features = outputs['person_reid']['features']
            reid_labels = targets['person_reid']
            reid_loss = self.reid_loss(reid_features, reid_labels)
            
            # Classification loss for training
            reid_logits = outputs['person_reid']['logits']
            cls_loss = self.ce_loss(reid_logits, reid_labels)
            
            losses['person_reid'] = reid_loss + cls_loss
            total_loss += self.task_weights[0] * losses['person_reid']
        
        # Presence detection loss
        if 'presence' in outputs and 'presence' in targets:
            losses['presence'] = self.ce_loss(outputs['presence'], targets['presence'])
            total_loss += self.task_weights[1] * losses['presence']
        
        # Gesture recognition loss
        if 'gesture' in outputs and 'gesture' in targets:
            losses['gesture'] = self.ce_loss(outputs['gesture'], targets['gesture'])
            total_loss += self.task_weights[2] * losses['gesture']
        
        # Fall detection loss
        if 'fall' in outputs and 'fall' in targets:
            losses['fall'] = self.ce_loss(outputs['fall'], targets['fall'])
            total_loss += self.task_weights[3] * losses['fall']
        
        # Object detection loss (simplified)
        if 'object_detection' in outputs and 'object_detection' in targets:
            # Implement YOLO-style loss here
            losses['object_detection'] = self.mse_loss(outputs['object_detection'], 
                                                      targets['object_detection'])
            total_loss += self.task_weights[4] * losses['object_detection']
        
        # Segmentation loss
        if 'segmentation' in outputs and 'segmentation' in targets:
            losses['segmentation'] = self.ce_loss(outputs['segmentation'], 
                                                 targets['segmentation'])
            total_loss += self.task_weights[5] * losses['segmentation']
        
        # Hazard detection loss
        if 'hazard' in outputs and 'hazard' in targets:
            losses['hazard'] = self.bce_loss(outputs['hazard'], 
                                           targets['hazard'].float())
            total_loss += self.task_weights[6] * losses['hazard']
        
        # Elevator detection loss
        if 'elevator' in outputs and 'elevator' in targets:
            losses['elevator'] = self.ce_loss(outputs['elevator'], targets['elevator'])
            total_loss += self.task_weights[7] * losses['elevator']
        
        # Compartment state loss
        if 'compartment' in outputs and 'compartment' in targets:
            losses['compartment'] = self.bce_loss(outputs['compartment'], 
                                                targets['compartment'].float())
            total_loss += self.task_weights[8] * losses['compartment']
        
        losses['total'] = total_loss
        return losses


def create_model(config=None):
    """Factory function to create the model"""
    if config is None:
        config = {
            'num_person_ids': 1000,
            'input_size': (224, 224),
            'num_classes_gesture': 4,
            'num_objects': 20,
            'num_segments': 10
        }
    
    return ElderAssistanceCNN(**config)


if __name__ == "__main__":
    # Test the model
    model = create_model()
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    outputs = model(x)
    
    print("Model outputs:")
    for task, output in outputs.items():
        if isinstance(output, dict):
            for key, value in output.items():
                print(f"{task}_{key}: {value.shape}")
        else:
            print(f"{task}: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")