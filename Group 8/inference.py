import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import time
import logging

from elder_assistance_cnn import ElderAssistanceCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElderAssistanceInference:
    """Inference engine for Elder Assistance CNN"""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class mappings
        self.class_mappings = self._get_class_mappings()
        
        # Person re-id database
        self.person_database = {}
        self.similarity_threshold = 0.7
        
        logger.info(f"Model loaded on {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> ElderAssistanceCNN:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint.get('model_config', {
                'num_person_ids': 1000,
                'input_size': (224, 224),
                'num_classes_gesture': 4,
                'num_objects': 20,
                'num_segments': 10
            })
        else:
            # Default config
            model_config = {
                'num_person_ids': 1000,
                'input_size': (224, 224),
                'num_classes_gesture': 4,
                'num_objects': 20,
                'num_segments': 10
            }
        
        model = ElderAssistanceCNN(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _get_class_mappings(self) -> Dict:
        """Define class mappings for different tasks"""
        return {
            'gesture': {
                0: 'None',
                1: 'Stop',
                2: 'Come',
                3: 'Other'
            },
            'fall': {
                0: 'Normal',
                1: 'Sitting',
                2: 'Fallen'
            },
            'presence': {
                0: 'Not Present',
                1: 'Present'
            },
            'elevator': {
                0: 'Elevator Open',
                1: 'Elevator Closed',
                2: 'Escalator',
                3: 'Neither'
            },
            'hazards': [
                'Stairs', 'Ramp', 'Curb', 'Wet Surface', 'Narrow Passage', 'Safe'
            ],
            'compartment': [
                'Door Open', 'Door Closed', 'Hands Detected'
            ],
            'segments': [
                'Background', 'Walkable', 'Stairs', 'Ramp', 'Curb', 
                'Wet Surface', 'Obstacle', 'Person', 'Vehicle', 'Unknown'
            ]
        }
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """Preprocess input image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image: Union[np.ndarray, Image.Image, str], 
                tasks: List[str] = None) -> Dict:
        """Run inference on image for specified tasks"""
        
        if tasks is None:
            tasks = ['all']
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            if len(tasks) == 1:
                outputs = self.model(image_tensor, task=tasks[0])
            else:
                outputs = self.model(image_tensor, task='all')
        
        inference_time = time.time() - start_time
        
        # Process outputs
        results = self._process_outputs(outputs)
        results['inference_time'] = inference_time
        
        return results
    
    def _process_outputs(self, outputs: Dict) -> Dict:
        """Process model outputs into interpretable results"""
        results = {}
        
        # Person Re-ID
        if 'person_reid' in outputs:
            features = outputs['person_reid']['features'].cpu().numpy()[0]
            logits = outputs['person_reid']['logits'].cpu().numpy()[0]
            
            results['person_reid'] = {
                'features': features,
                'confidence': float(np.max(F.softmax(torch.tensor(logits), dim=0).numpy())),
                'predicted_id': int(np.argmax(logits))
            }
        
        # Presence Detection
        if 'presence' in outputs:
            probs = F.softmax(outputs['presence'], dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            
            results['presence'] = {
                'prediction': self.class_mappings['presence'][pred],
                'confidence': float(probs[pred]),
                'probabilities': {
                    self.class_mappings['presence'][i]: float(probs[i]) 
                    for i in range(len(probs))
                }
            }
        
        # Gesture Recognition
        if 'gesture' in outputs:
            probs = F.softmax(outputs['gesture'], dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            
            results['gesture'] = {
                'prediction': self.class_mappings['gesture'][pred],
                'confidence': float(probs[pred]),
                'probabilities': {
                    self.class_mappings['gesture'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            }
        
        # Fall Detection
        if 'fall' in outputs:
            probs = F.softmax(outputs['fall'], dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            
            results['fall'] = {
                'prediction': self.class_mappings['fall'][pred],
                'confidence': float(probs[pred]),
                'probabilities': {
                    self.class_mappings['fall'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            }
        
        # Hazard Detection (multi-label)
        if 'hazard' in outputs:
            probs = torch.sigmoid(outputs['hazard']).cpu().numpy()[0]
            
            results['hazards'] = {
                'predictions': [
                    self.class_mappings['hazards'][i] 
                    for i, prob in enumerate(probs) if prob > 0.5
                ],
                'probabilities': {
                    self.class_mappings['hazards'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            }
        
        # Elevator Detection
        if 'elevator' in outputs:
            probs = F.softmax(outputs['elevator'], dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            
            results['elevator'] = {
                'prediction': self.class_mappings['elevator'][pred],
                'confidence': float(probs[pred]),
                'probabilities': {
                    self.class_mappings['elevator'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            }
        
        # Compartment State (multi-label)
        if 'compartment' in outputs:
            probs = torch.sigmoid(outputs['compartment']).cpu().numpy()[0]
            
            results['compartment'] = {
                'predictions': [
                    self.class_mappings['compartment'][i]
                    for i, prob in enumerate(probs) if prob > 0.5
                ],
                'probabilities': {
                    self.class_mappings['compartment'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            }
        
        # Semantic Segmentation
        if 'segmentation' in outputs:
            seg_map = torch.argmax(outputs['segmentation'], dim=1).cpu().numpy()[0]
            
            results['segmentation'] = {
                'map': seg_map,
                'walkable_percentage': float(
                    np.sum(seg_map == 1) / (seg_map.shape[0] * seg_map.shape[1]) * 100
                ),
                'hazard_pixels': {
                    segment: int(np.sum(seg_map == i))
                    for i, segment in enumerate(self.class_mappings['segments'])
                }
            }
        
        # Object Detection (simplified)
        if 'object_detection' in outputs:
            # This would need proper post-processing for actual object detection
            results['object_detection'] = {
                'raw_output_shape': outputs['object_detection'].shape,
                'note': 'Object detection requires proper post-processing implementation'
            }
        
        return results
    
    def register_person(self, person_id: str, image: Union[np.ndarray, Image.Image, str]):
        """Register a person in the Re-ID database"""
        results = self.predict(image, tasks=['person_reid'])
        
        if 'person_reid' in results:
            features = results['person_reid']['features']
            self.person_database[person_id] = features
            logger.info(f"Registered person {person_id} in database")
        else:
            logger.error("Failed to extract person features")
    
    def identify_person(self, image: Union[np.ndarray, Image.Image, str]) -> Optional[str]:
        """Identify person using Re-ID database"""
        if not self.person_database:
            logger.warning("Person database is empty")
            return None
        
        results = self.predict(image, tasks=['person_reid'])
        
        if 'person_reid' not in results:
            return None
        
        query_features = results['person_reid']['features']
        
        # Compute similarities
        similarities = {}
        for person_id, db_features in self.person_database.items():
            similarity = np.dot(query_features, db_features) / (
                np.linalg.norm(query_features) * np.linalg.norm(db_features)
            )
            similarities[person_id] = similarity
        
        # Find best match
        best_match = max(similarities, key=similarities.get)
        best_similarity = similarities[best_match]
        
        if best_similarity > self.similarity_threshold:
            return best_match
        else:
            return None
    
    def analyze_scene_safety(self, image: Union[np.ndarray, Image.Image, str]) -> Dict:
        """Comprehensive scene safety analysis"""
        results = self.predict(image, tasks=['hazards', 'segmentation', 'fall'])
        
        safety_score = 100.0  # Start with perfect safety
        warnings = []
        
        # Check hazards
        if 'hazards' in results:
            detected_hazards = results['hazards']['predictions']
            if detected_hazards and 'Safe' not in detected_hazards:
                safety_score -= len(detected_hazards) * 15
                warnings.extend([f"Hazard detected: {h}" for h in detected_hazards])
        
        # Check segmentation for walkable area
        if 'segmentation' in results:
            walkable_pct = results['segmentation']['walkable_percentage']
            if walkable_pct < 50:
                safety_score -= 20
                warnings.append(f"Limited walkable area: {walkable_pct:.1f}%")
        
        # Check for fall detection
        if 'fall' in results:
            fall_state = results['fall']['prediction']
            if fall_state != 'Normal':
                safety_score -= 50
                warnings.append(f"Fall detected: {fall_state}")
        
        safety_score = max(0, safety_score)  # Ensure non-negative
        
        return {
            'safety_score': safety_score,
            'safety_level': self._get_safety_level(safety_score),
            'warnings': warnings,
            'detailed_results': results
        }
    
    def _get_safety_level(self, score: float) -> str:
        """Convert safety score to level"""
        if score >= 80:
            return "Safe"
        elif score >= 60:
            return "Caution"
        elif score >= 40:
            return "Warning"
        else:
            return "Dangerous"
    
    def batch_predict(self, images: List[Union[np.ndarray, Image.Image, str]], 
                     tasks: List[str] = None) -> List[Dict]:
        """Run inference on multiple images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.predict(image, tasks)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                results.append({'image_index': i, 'error': str(e)})
        
        return results


def main():
    """Example usage"""
    
    # Initialize inference engine
    inference = ElderAssistanceInference('checkpoints/best.pth')
    
    # Example: Process single image
    image_path = 'example_image.jpg'
    
    if os.path.exists(image_path):
        # Full analysis
        results = inference.predict(image_path)
        print("Full Analysis Results:")
        print(json.dumps(results, indent=2, default=str))
        
        # Scene safety analysis
        safety_results = inference.analyze_scene_safety(image_path)
        print("\nSafety Analysis:")
        print(json.dumps(safety_results, indent=2, default=str))
        
        # Person registration example
        inference.register_person('elder_001', image_path)
        
        # Person identification
        identified = inference.identify_person(image_path)
        print(f"\nIdentified person: {identified}")
    
    else:
        print(f"Example image not found: {image_path}")
        print("Please provide a valid image path to test the inference engine")


if __name__ == "__main__":
    main()