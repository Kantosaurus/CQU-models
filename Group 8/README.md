# Elder Assistance CNN Model

A comprehensive multi-task CNN for elder assistance robots, addressing all critical perception and safety requirements.

## Features

### User & Interaction
- **Person Re-ID / Tracking**: Lock onto correct elder in crowds and re-acquire if occluded
- **User Presence Verification**: Short-range face/torso confirmation before resume or opening compartments
- **Gesture Commands**: Recognize "Stop" (raised palm) and "Come" (beckon) gestures
- **Fall Detection**: Detect if user collapses or sits abruptly

### Scene Understanding & Safety
- **Object Detection**: People, bikes/scooters, strollers, pets, carts, vehicles, low obstacles
- **Semantic Segmentation**: Classify walkable surfaces vs. stairs, ramps, curbs, wet/slippery tiles
- **Hazard Detection**: Stair edge distance, slope cues, puddles/spills, narrow passages
- **Crowd Density Analysis**: Prefer low-congestion paths and yield behavior

### Task-Specific Perception
- **Elevator/Escalator Recognition**: Identify elevator doors, out-of-service signage, avoid escalators
- **Compartment State Monitoring**: Door status, hand detection, load placement/removal
- **Payload Assessment**: Rough fill level and large object detection

## Model Architecture

The model uses a multi-task learning approach with:
- **Backbone**: EfficientNet-B4 for feature extraction
- **Feature Pyramid Network**: Multi-scale feature processing
- **Task-specific Heads**: Specialized outputs for each perception task
- **Adaptive Loss Weighting**: Learnable task importance balancing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from training_pipeline import main
from elder_assistance_cnn import create_model

# Configure your dataset path
config = {
    'data_root': 'path/to/your/dataset',
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'save_dir': 'checkpoints',
    'num_workers': 4
}

# Start training
main()
```

### Inference

```python
from inference import ElderAssistanceInference

# Initialize inference engine
inference = ElderAssistanceInference('checkpoints/best.pth')

# Single image prediction
results = inference.predict('path/to/image.jpg')

# Scene safety analysis
safety_results = inference.analyze_scene_safety('path/to/image.jpg')

# Person registration and identification
inference.register_person('elder_001', 'path/to/person_image.jpg')
identified_person = inference.identify_person('path/to/query_image.jpg')
```

## Dataset Format

Expected directory structure:

```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── annotations/
│   ├── person_reid.json
│   ├── presence.json
│   ├── gestures.json
│   ├── fall_detection.json
│   ├── objects.json
│   ├── segmentation/
│   │   ├── mask1.png
│   │   ├── mask2.png
│   │   └── ...
│   ├── hazards.json
│   ├── elevator.json
│   └── compartment.json
```

### Annotation Formats

#### person_reid.json
```json
{
    "image1": {"person_id": 0},
    "image2": {"person_id": 1}
}
```

#### presence.json
```json
{
    "image1": {"present": 1},
    "image2": {"present": 0}
}
```

#### gestures.json
```json
{
    "image1": {"gesture_class": 0},  // 0: None, 1: Stop, 2: Come, 3: Other
    "image2": {"gesture_class": 1}
}
```

#### fall_detection.json
```json
{
    "image1": {"fall_state": 0},  // 0: Normal, 1: Sitting, 2: Fallen
    "image2": {"fall_state": 2}
}
```

#### hazards.json
```json
{
    "image1": {"hazards": [0, 0, 1, 0, 0, 1]},  // Multi-label: [stairs, ramp, curb, wet, narrow, safe]
    "image2": {"hazards": [0, 0, 0, 0, 0, 1]}
}
```

#### elevator.json
```json
{
    "image1": {"elevator_state": 0},  // 0: Open, 1: Closed, 2: Escalator, 3: Neither
    "image2": {"elevator_state": 3}
}
```

#### compartment.json
```json
{
    "image1": {"states": [1, 0, 0]},  // Multi-label: [door_open, door_closed, hands_detected]
    "image2": {"states": [0, 1, 1]}
}
```

## Model Outputs

The model returns a dictionary with the following structure:

```python
{
    'person_reid': {
        'features': np.array,      # 256-dim normalized features
        'confidence': float,       # Classification confidence
        'predicted_id': int        # Person ID (for training)
    },
    'presence': {
        'prediction': str,         # "Present" or "Not Present"
        'confidence': float,       # Prediction confidence
        'probabilities': dict      # All class probabilities
    },
    'gesture': {
        'prediction': str,         # "None", "Stop", "Come", "Other"
        'confidence': float,       # Prediction confidence
        'probabilities': dict      # All class probabilities
    },
    'fall': {
        'prediction': str,         # "Normal", "Sitting", "Fallen"
        'confidence': float,       # Prediction confidence
        'probabilities': dict      # All class probabilities
    },
    'hazards': {
        'predictions': list,       # Active hazards
        'probabilities': dict      # All hazard probabilities
    },
    'elevator': {
        'prediction': str,         # Elevator/escalator state
        'confidence': float,       # Prediction confidence
        'probabilities': dict      # All class probabilities
    },
    'compartment': {
        'predictions': list,       # Active states
        'probabilities': dict      # All state probabilities
    },
    'segmentation': {
        'map': np.array,          # Pixel-wise segmentation map
        'walkable_percentage': float,  # % of walkable pixels
        'hazard_pixels': dict     # Pixel counts per segment type
    }
}
```

## Performance Considerations

- **Model Size**: ~90M parameters (EfficientNet-B4 backbone)
- **Inference Speed**: ~50ms on modern GPU, ~200ms on CPU
- **Memory Usage**: ~2GB GPU memory for training, ~500MB for inference
- **Input Resolution**: 224x224 (configurable)

## Safety Features

The model includes several safety-critical features:

1. **Multi-level Fall Detection**: Distinguishes between normal, sitting, and fallen states
2. **Comprehensive Hazard Assessment**: Identifies multiple environmental hazards
3. **Person Tracking Reliability**: Robust re-identification even with occlusions
4. **Real-time Scene Understanding**: Fast enough for real-time robot navigation

## Customization

To adapt the model for your specific use case:

1. **Modify Class Mappings**: Update the number of classes in each task
2. **Add New Tasks**: Extend the model architecture with additional heads
3. **Adjust Input Size**: Change image resolution for speed/accuracy trade-offs
4. **Fine-tune Backbone**: Use different backbone architectures (ResNet, Vision Transformer, etc.)

## Training Tips

1. **Data Balance**: Ensure balanced representation across all tasks
2. **Augmentation**: Use appropriate augmentations for robotic vision scenarios
3. **Transfer Learning**: Start with ImageNet-pretrained backbone
4. **Multi-task Weighting**: Monitor and adjust task loss weights during training
5. **Validation Strategy**: Use temporally separated train/val splits for realistic evaluation

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{elder_assistance_cnn,
  title={Multi-task CNN for Elder Assistance Robot Perception},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```