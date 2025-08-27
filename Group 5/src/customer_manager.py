import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import face_recognition
from datetime import datetime

class Customer:
    def __init__(self, customer_id: str, face_encoding: np.ndarray, name: str = ""):
        self.customer_id = customer_id
        self.face_encoding = face_encoding
        self.name = name
        self.shopping_history = []
        self.preferences = {}
        self.created_date = datetime.now()
        self.last_visit = datetime.now()

    def add_purchase(self, product_ids: List[str]):
        purchase_record = {
            'date': datetime.now(),
            'products': product_ids
        }
        self.shopping_history.append(purchase_record)
        self.last_visit = datetime.now()

    def get_purchase_frequency(self, product_id: str) -> int:
        count = 0
        for record in self.shopping_history:
            if product_id in record['products']:
                count += 1
        return count

    def get_category_preferences(self) -> Dict[str, int]:
        category_counts = {}
        for record in self.shopping_history:
            for product_id in record['products']:
                if hasattr(self, '_product_manager'):
                    product = self._product_manager.get_product(product_id)
                    if product:
                        category = product.category
                        category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts

class CustomerManager:
    def __init__(self, data_file: str = "data/customers.pkl"):
        self.data_file = data_file
        self.customers: Dict[str, Customer] = {}
        self.face_encodings: List[np.ndarray] = []
        self.customer_ids: List[str] = []
        self.load_customers()

    def load_customers(self):
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.customers = data.get('customers', {})
                self.face_encodings = []
                self.customer_ids = []
                for customer_id, customer in self.customers.items():
                    self.face_encodings.append(customer.face_encoding)
                    self.customer_ids.append(customer_id)
        except (FileNotFoundError, EOFError):
            print("No existing customer data found. Starting fresh.")

    def save_customers(self):
        import os
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'wb') as f:
            pickle.dump({'customers': self.customers}, f)

    def register_customer(self, face_image: np.ndarray, customer_id: str = None, name: str = "") -> str:
        face_encodings = face_recognition.face_encodings(face_image)
        if not face_encodings:
            raise ValueError("No face detected in image")
        
        face_encoding = face_encodings[0]
        
        if customer_id is None:
            customer_id = f"customer_{len(self.customers) + 1:04d}"
        
        customer = Customer(customer_id, face_encoding, name)
        self.customers[customer_id] = customer
        self.face_encodings.append(face_encoding)
        self.customer_ids.append(customer_id)
        self.save_customers()
        
        return customer_id

    def recognize_customer(self, face_image: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        if not self.face_encodings:
            return None
            
        face_encodings = face_recognition.face_encodings(face_image)
        if not face_encodings:
            return None
        
        unknown_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(self.face_encodings, unknown_encoding, tolerance)
        
        if True in matches:
            match_index = matches.index(True)
            return self.customer_ids[match_index]
        
        return None

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        return self.customers.get(customer_id)

    def add_purchase_history(self, customer_id: str, product_ids: List[str]):
        customer = self.get_customer(customer_id)
        if customer:
            customer.add_purchase(product_ids)
            self.save_customers()

    def is_new_customer(self, customer_id: str) -> bool:
        customer = self.get_customer(customer_id)
        return customer and len(customer.shopping_history) == 0

    def get_customer_stats(self) -> Dict[str, int]:
        return {
            'total_customers': len(self.customers),
            'new_customers': sum(1 for c in self.customers.values() if len(c.shopping_history) == 0),
            'returning_customers': sum(1 for c in self.customers.values() if len(c.shopping_history) > 0)
        }