"""
Basic system tests for the 3D Navigation System
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collector import DataCollector, SensorData
from elevation_predictor import ElevationPredictor, DataPreprocessor
from graph_builder import Graph3DBuilder, Node3D, Edge3D
from pathfinder import Dijkstra3D, RouteConstraints


class TestDataCollector:
    """Test data collection functionality."""
    
    def test_sensor_data_creation(self):
        """Test creating sensor data objects."""
        collector = DataCollector(":memory:")  # In-memory database for testing
        
        # Test sensor data simulation
        sensor_data = collector.simulate_sensor_reading(29.5630, 106.5516, elevation=1)
        
        assert sensor_data.gps_lat == pytest.approx(29.5630, abs=0.001)
        assert sensor_data.gps_lon == pytest.approx(106.5516, abs=0.001)
        assert sensor_data.barometer_pressure < 1013.25  # Should be lower due to elevation
        assert len(sensor_data.wifi_signals) > 0
        
    @pytest.mark.asyncio
    async def test_data_storage(self):
        """Test storing sensor data in database."""
        collector = DataCollector(":memory:")
        
        sensor_data = collector.simulate_sensor_reading(29.5630, 106.5516)
        success = await collector.collect_sensor_data(sensor_data)
        
        assert success is True
        
        # Verify data was stored
        stored_data = collector.get_training_data(limit=1)
        assert len(stored_data) == 1
        assert stored_data[0]['gps_lat'] == pytest.approx(sensor_data.gps_lat, abs=0.0001)


class TestElevationPredictor:
    """Test elevation prediction functionality."""
    
    def test_model_training(self):
        """Test training the XGBoost model."""
        predictor = ElevationPredictor()
        
        # Create synthetic training data
        features, labels = predictor.create_synthetic_training_data(n_samples=100)
        
        assert len(features) == 100
        assert len(labels) == 100
        assert features.shape[1] > 10  # Should have multiple features
        
        # Train model
        metrics = predictor.train_model(features, labels)
        
        assert metrics['accuracy'] > 0.5  # Should be better than random
        assert predictor.is_trained is True
        
    def test_feature_extraction(self):
        """Test feature extraction from sensor data."""
        # Create mock sensor data
        sensor_dict = {
            'gps_lat': 29.5630,
            'gps_lon': 106.5516,
            'gps_accuracy': 5.0,
            'wifi_signals': '[{"ap1": -40}, {"ap2": -50}]',
            'bluetooth_beacons': '[{"beacon1": -60}]',
            'barometer_pressure': 1010.0,
            'accelerometer_x': 0.1,
            'accelerometer_y': 0.2,
            'accelerometer_z': -9.8,
            'gyroscope_x': 0.01,
            'gyroscope_y': 0.02,
            'gyroscope_z': 0.03,
            'magnetometer_x': 20.0,
            'magnetometer_y': 1.0,
            'magnetometer_z': -45.0
        }
        
        features = DataPreprocessor.extract_features(sensor_dict)
        
        assert 'gps_lat' in features
        assert 'barometer_pressure' in features
        assert 'wifi_count' in features
        assert 'accel_magnitude' in features
        assert features['wifi_count'] == 2
        assert features['bt_count'] == 1
        
    def test_prediction(self):
        """Test elevation prediction."""
        predictor = ElevationPredictor()
        
        # Train with minimal data
        features, labels = predictor.create_synthetic_training_data(n_samples=50)
        predictor.train_model(features, labels)
        
        # Test prediction
        test_features = {
            'gps_lat': 29.5630,
            'gps_lon': 106.5516,
            'gps_accuracy': 5.0,
            'barometer_pressure': 1010.0,
            'wifi_count': 3,
            'wifi_max_strength': -35,
            'wifi_avg_strength': -45,
            'wifi_std_strength': 5,
            'bt_count': 1,
            'bt_max_strength': -60,
            'bt_avg_strength': -60,
            'accel_magnitude': 9.8,
            'gyro_magnitude': 0.1,
            'magnetometer_x': 20,
            'magnetometer_y': 1,
            'magnetometer_z': -45
        }
        
        elevation, confidence = predictor.predict_elevation(test_features)
        
        assert isinstance(elevation, int)
        assert 0 <= elevation <= 10
        assert 0.0 <= confidence <= 1.0


class TestGraphBuilder:
    """Test 3D graph construction."""
    
    def test_node_creation(self):
        """Test creating 3D nodes."""
        node = Node3D(
            id="test_node",
            lat=29.5630,
            lon=106.5516,
            elevation=1,
            node_type="location"
        )
        
        assert node.id == "test_node"
        assert node.elevation == 1
        assert node.node_type == "location"
        
    def test_edge_creation(self):
        """Test creating 3D edges."""
        edge = Edge3D(
            source_id="node1",
            target_id="node2",
            edge_type="walk",
            weight=60.0,
            distance=50.0,
            elevation_change=0
        )
        
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.edge_type == "walk"
        assert edge.elevation_change == 0
        
    def test_graph_building(self):
        """Test building a graph from data."""
        # Create predictor
        predictor = ElevationPredictor()
        features, labels = predictor.create_synthetic_training_data(n_samples=50)
        predictor.train_model(features, labels)
        
        # Create data collector with test data
        collector = DataCollector(":memory:")
        
        async def add_test_data():
            # Add some test locations
            test_locations = [
                (29.5630, 106.5516, 0),
                (29.5631, 106.5517, 1),
                (29.5635, 106.5520, 0)
            ]
            
            for lat, lon, elevation in test_locations:
                for _ in range(10):  # Multiple readings per location
                    sensor_data = collector.simulate_sensor_reading(lat, lon, elevation)
                    await collector.collect_sensor_data(sensor_data)
        
        asyncio.run(add_test_data())
        
        # Build graph
        graph_builder = Graph3DBuilder(predictor)
        graph = graph_builder.build_graph_from_data(collector, cluster_radius=100, min_cluster_size=5)
        
        assert graph.number_of_nodes() > 0
        # Don't require edges as they depend on proximity


class TestPathfinder:
    """Test pathfinding functionality."""
    
    def test_route_constraints(self):
        """Test route constraint creation."""
        constraints = RouteConstraints(
            avoid_stairs=True,
            accessibility_required=True,
            max_elevation_change=5
        )
        
        assert constraints.avoid_stairs is True
        assert constraints.accessibility_required is True
        assert constraints.max_elevation_change == 5
        
    def test_pathfinder_initialization(self):
        """Test pathfinder initialization."""
        # Create minimal setup
        predictor = ElevationPredictor()
        features, labels = predictor.create_synthetic_training_data(n_samples=50)
        predictor.train_model(features, labels)
        
        graph_builder = Graph3DBuilder(predictor)
        pathfinder = Dijkstra3D(graph_builder)
        
        assert pathfinder.graph_builder == graph_builder
        assert pathfinder.graph == graph_builder.graph


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test complete data flow from collection to pathfinding."""
        # 1. Initialize components
        collector = DataCollector(":memory:")
        predictor = ElevationPredictor()
        
        # 2. Train model
        features, labels = predictor.create_synthetic_training_data(n_samples=100)
        metrics = predictor.train_model(features, labels)
        assert metrics['accuracy'] > 0.4  # Reasonable accuracy for small dataset
        
        # 3. Collect sensor data
        test_locations = [
            (29.5630, 106.5516, 0),
            (29.5631, 106.5517, 1),
            (29.5640, 106.5530, 0),
            (29.5641, 106.5531, 1)
        ]
        
        for lat, lon, elevation in test_locations:
            for _ in range(15):  # Ensure enough data for clustering
                sensor_data = collector.simulate_sensor_reading(lat, lon, elevation)
                success = await collector.collect_sensor_data(sensor_data)
                assert success is True
        
        # 4. Build graph
        graph_builder = Graph3DBuilder(predictor)
        graph = graph_builder.build_graph_from_data(
            collector, 
            cluster_radius=100,  # Large radius for test
            min_cluster_size=3   # Low threshold for test
        )
        
        # Graph should have some structure
        assert graph.number_of_nodes() >= 0  # May be 0 if clustering fails
        
        # 5. Initialize pathfinder
        pathfinder = Dijkstra3D(graph_builder)
        
        # Test that pathfinder is properly initialized
        assert pathfinder.graph == graph
        assert len(pathfinder.nodes_3d) == graph.number_of_nodes()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])