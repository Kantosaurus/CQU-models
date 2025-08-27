import cv2
import numpy as np
import face_recognition
from typing import Optional, Tuple, List
import logging
from datetime import datetime

class FaceRecognitionService:
    """
    Service for handling face recognition functionality for the smart cart robot.
    Integrates with customer management system.
    """
    
    def __init__(self, customer_manager=None):
        self.customer_manager = customer_manager
        self.logger = logging.getLogger(__name__)
        
        # Face detection parameters
        self.face_detection_model = 'hog'  # 'hog' for CPU, 'cnn' for GPU
        self.recognition_tolerance = 0.6
        self.min_face_size = 50  # Minimum face size in pixels
        
        # Initialize camera if available
        self.camera = None
        self.initialize_camera()
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize camera for face recognition"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                self.logger.warning(f"Could not open camera {camera_index}")
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera"""
        if not self.camera or not self.camera.isOpened():
            self.logger.error("Camera not available")
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("Failed to capture frame")
            return None
        
        # Convert BGR to RGB (face_recognition expects RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        try:
            face_locations = face_recognition.face_locations(
                image, model=self.face_detection_model)
            
            # Filter out faces that are too small
            filtered_faces = []
            for (top, right, bottom, left) in face_locations:
                width = right - left
                height = bottom - top
                if width >= self.min_face_size and height >= self.min_face_size:
                    filtered_faces.append((top, right, bottom, left))
            
            return filtered_faces
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face_encoding(self, image: np.ndarray, 
                             face_location: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Extract face encoding from an image.
        
        Args:
            image: RGB image array
            face_location: Optional face location tuple (top, right, bottom, left)
        
        Returns:
            Face encoding array or None if extraction fails
        """
        try:
            if face_location:
                encodings = face_recognition.face_encodings(image, [face_location])
            else:
                encodings = face_recognition.face_encodings(image)
            
            if encodings:
                return encodings[0]
            else:
                self.logger.warning("No face encoding could be extracted")
                return None
        except Exception as e:
            self.logger.error(f"Face encoding extraction failed: {e}")
            return None
    
    def recognize_customer(self, image: np.ndarray) -> Optional[str]:
        """
        Recognize a customer from their face.
        
        Args:
            image: RGB image array containing the customer's face
            
        Returns:
            Customer ID if recognized, None otherwise
        """
        if not self.customer_manager:
            self.logger.error("Customer manager not available")
            return None
        
        try:
            # Extract face encoding from the image
            face_encoding = self.extract_face_encoding(image)
            if face_encoding is None:
                return None
            
            # Use customer manager's recognition method
            customer_id = self.customer_manager.recognize_customer(
                image, tolerance=self.recognition_tolerance)
            
            if customer_id:
                self.logger.info(f"Customer recognized: {customer_id}")
            else:
                self.logger.info("Customer not recognized")
            
            return customer_id
        except Exception as e:
            self.logger.error(f"Customer recognition failed: {e}")
            return None
    
    def register_new_customer(self, image: np.ndarray, 
                            customer_name: str = "") -> Optional[str]:
        """
        Register a new customer using their face.
        
        Args:
            image: RGB image array containing the customer's face
            customer_name: Optional customer name
            
        Returns:
            New customer ID if successful, None otherwise
        """
        if not self.customer_manager:
            self.logger.error("Customer manager not available")
            return None
        
        try:
            # Check if face is detected
            face_locations = self.detect_faces(image)
            if not face_locations:
                self.logger.error("No face detected for registration")
                return None
            
            # Register with customer manager
            customer_id = self.customer_manager.register_customer(
                image, name=customer_name)
            
            self.logger.info(f"New customer registered: {customer_id}")
            return customer_id
        except Exception as e:
            self.logger.error(f"Customer registration failed: {e}")
            return None
    
    def process_customer_recognition(self) -> Optional[str]:
        """
        Capture frame and attempt customer recognition.
        
        Returns:
            Customer ID if recognized, None otherwise
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        return self.recognize_customer(frame)
    
    def interactive_customer_recognition(self, timeout_seconds: int = 30) -> Optional[str]:
        """
        Interactive customer recognition with visual feedback.
        
        Args:
            timeout_seconds: Maximum time to wait for recognition
            
        Returns:
            Customer ID if recognized, None if timeout or error
        """
        if not self.camera or not self.camera.isOpened():
            self.logger.error("Camera not available for interactive recognition")
            return None
        
        start_time = datetime.now()
        recognized_customer = None
        
        print("Looking for customer... Press 'q' to quit or 'r' to register new customer")
        
        while (datetime.now() - start_time).seconds < timeout_seconds:
            frame = self.capture_frame()
            if frame is None:
                continue
            
            # Convert RGB back to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect faces for visual feedback
            face_locations = self.detect_faces(frame)
            
            # Draw rectangles around detected faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Try recognition
            customer_id = self.recognize_customer(frame)
            if customer_id:
                cv2.putText(display_frame, f"Welcome back, {customer_id}!", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                recognized_customer = customer_id
                break
            elif face_locations:
                cv2.putText(display_frame, "Face detected - Processing...", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Please position your face in view", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display the frame
            cv2.imshow('Smart Cart - Customer Recognition', display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and face_locations:
                # Register new customer
                customer_name = input("Enter customer name (optional): ")
                new_customer_id = self.register_new_customer(frame, customer_name)
                if new_customer_id:
                    recognized_customer = new_customer_id
                    break
        
        cv2.destroyAllWindows()
        return recognized_customer
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

# Voice recognition integration placeholder
class VoiceRecognitionService:
    """
    Placeholder for voice recognition functionality.
    In a full implementation, this would integrate with speech-to-text services.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wake_words = ["hello", "smart cart", "help me"]
    
    def listen_for_wake_word(self, timeout_seconds: int = 10) -> bool:
        """
        Listen for wake words to activate the cart.
        
        Returns:
            True if wake word detected, False otherwise
        """
        # Placeholder implementation
        self.logger.info("Listening for wake word...")
        print("Voice recognition placeholder - say 'hello' to activate")
        
        # In real implementation, this would use speech recognition
        user_input = input("Simulate voice input (type 'hello'): ")
        return user_input.lower() in self.wake_words
    
    def process_voice_command(self) -> Optional[str]:
        """
        Process voice commands for product searches or cart operations.
        
        Returns:
            Processed command text or None
        """
        # Placeholder implementation
        command = input("Voice command (e.g., 'find milk', 'show recommendations'): ")
        return command if command else None