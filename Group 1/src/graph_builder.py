"""
3D Graph Construction System
Builds elevation-aware graphs from crowdsourced location data with (x, y, z) coordinates.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict
import json
import sqlite3
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_collector import DataCollector
from elevation_predictor import ElevationPredictor, DataPreprocessor


@dataclass
class Node3D:
    """3D graph node with elevation information."""
    id: str
    lat: float
    lon: float
    elevation: int
    node_type: str  # 'location', 'junction', 'entrance', 'exit'
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Edge3D:
    """3D graph edge representing walkable/driveable connections."""
    source_id: str
    target_id: str
    edge_type: str  # 'walk', 'stairs', 'escalator', 'elevator', 'ramp', 'tunnel'
    weight: float  # Travel time/effort cost
    distance: float  # Physical distance in meters
    elevation_change: int  # Change in elevation levels
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class Graph3DBuilder:
    """Builds 3D navigation graphs from sensor data and elevation predictions."""
    
    def __init__(self, elevation_predictor: ElevationPredictor):
        self.elevation_predictor = elevation_predictor
        self.graph = nx.DiGraph()
        self.nodes_3d = {}  # node_id -> Node3D
        self.edges_3d = {}  # (source, target) -> Edge3D
        self.location_clusters = {}  # (lat, lon) -> cluster_info
        
    def build_graph_from_data(self, data_collector: DataCollector, 
                            cluster_radius: float = 50.0,
                            min_cluster_size: int = 5) -> nx.DiGraph:
        """
        Build 3D graph from collected sensor data.
        
        Args:
            data_collector: DataCollector with sensor readings
            cluster_radius: Radius for location clustering (meters)
            min_cluster_size: Minimum points needed to form a cluster
            
        Returns:
            NetworkX DiGraph with 3D nodes and edges
        """
        print("Building 3D graph from sensor data...")
        
        # Step 1: Get location clusters
        raw_clusters = data_collector.get_location_clusters(cluster_radius)
        print(f"Found {len(raw_clusters)} raw location clusters")
        
        # Step 2: Predict elevation for each cluster
        self._predict_cluster_elevations(data_collector, raw_clusters, min_cluster_size)
        
        # Step 3: Create nodes for each elevation at each location
        self._create_3d_nodes()
        
        # Step 4: Create edges between nodes
        self._create_3d_edges()
        
        # Step 5: Add graph to NetworkX
        self._build_networkx_graph()
        
        print(f"Built 3D graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _predict_cluster_elevations(self, data_collector: DataCollector, 
                                  raw_clusters: List[Tuple], min_cluster_size: int):
        """Predict elevation levels for location clusters."""
        print("Predicting elevations for location clusters...")
        
        raw_data = data_collector.get_training_data()
        
        for lat_cluster, lon_cluster, count in raw_clusters:
            if count < min_cluster_size:
                continue
                
            # Find all sensor readings near this cluster
            cluster_readings = []
            for record in raw_data:
                distance = geodesic((record['gps_lat'], record['gps_lon']), 
                                  (lat_cluster, lon_cluster)).meters
                if distance <= 100:  # Within 100m of cluster center
                    cluster_readings.append(record)
            
            if not cluster_readings:
                continue
            
            # Predict elevation for each reading
            elevations_found = set()
            elevation_predictions = {}
            
            for record in cluster_readings:
                features = DataPreprocessor.extract_features(record)
                try:
                    elevation, confidence = self.elevation_predictor.predict_elevation(features)
                    if confidence > 0.5:  # Only high-confidence predictions
                        elevations_found.add(elevation)
                        if elevation not in elevation_predictions:
                            elevation_predictions[elevation] = []
                        elevation_predictions[elevation].append({
                            'confidence': confidence,
                            'record': record
                        })
                except:
                    continue
            
            # Store cluster information
            if elevations_found:
                cluster_key = (round(lat_cluster, 6), round(lon_cluster, 6))
                self.location_clusters[cluster_key] = {
                    'elevations': sorted(elevations_found),
                    'predictions': elevation_predictions,
                    'total_readings': len(cluster_readings)
                }
    
    def _create_3d_nodes(self):
        """Create 3D nodes for each elevation level at each location."""
        print("Creating 3D nodes...")
        
        for (lat, lon), cluster_info in self.location_clusters.items():
            for elevation in cluster_info['elevations']:
                node_id = f"node_{lat}_{lon}_e{elevation}"
                
                # Determine node type based on elevation and context
                if elevation == 0:
                    node_type = 'entrance'  # Ground level entrances
                elif len(cluster_info['elevations']) > 1:
                    node_type = 'junction'  # Multi-level locations
                else:
                    node_type = 'location'  # Single-level locations
                
                # Calculate average confidence for this elevation
                predictions = cluster_info['predictions'].get(elevation, [])
                avg_confidence = np.mean([p['confidence'] for p in predictions]) if predictions else 0.5
                
                node = Node3D(
                    id=node_id,
                    lat=lat,
                    lon=lon,
                    elevation=elevation,
                    node_type=node_type,
                    properties={
                        'confidence': avg_confidence,
                        'readings_count': len(predictions),
                        'total_cluster_readings': cluster_info['total_readings']
                    }
                )
                
                self.nodes_3d[node_id] = node
    
    def _create_3d_edges(self):
        """Create edges between 3D nodes representing possible paths."""
        print("Creating 3D edges...")
        
        nodes_list = list(self.nodes_3d.values())
        
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                # Calculate geographical distance
                distance = geodesic((node1.lat, node1.lon), (node2.lat, node2.lon)).meters
                
                # Skip if too far apart (not walkable)
                if distance > 200:  # 200m max walking distance
                    continue
                
                elevation_diff = abs(node1.elevation - node2.elevation)
                
                # Same location, different elevations
                if distance < 10 and elevation_diff > 0:
                    self._add_vertical_connections(node1, node2, distance)
                
                # Different locations, same or adjacent elevations
                elif elevation_diff <= 1:
                    self._add_horizontal_connections(node1, node2, distance)
    
    def _add_vertical_connections(self, node1: Node3D, node2: Node3D, distance: float):
        """Add vertical connections (stairs, elevators, etc.) between elevation levels."""
        elevation_diff = abs(node1.elevation - node2.elevation)
        
        # Determine connection type and cost
        if elevation_diff == 1:
            # Single level difference - likely stairs or escalator
            edge_type = 'stairs'
            base_cost = 60  # 60 seconds to go up/down one level
        else:
            # Multi-level difference - likely elevator
            edge_type = 'elevator'
            base_cost = 30 + (elevation_diff * 15)  # Base time + time per floor
        
        # Add bidirectional edges
        for source, target in [(node1, node2), (node2, node1)]:
            edge_id = (source.id, target.id)
            
            # Going up costs more than going down
            if target.elevation > source.elevation:
                weight = base_cost * 1.2  # 20% penalty for going up
            else:
                weight = base_cost * 0.8  # 20% bonus for going down
            
            edge = Edge3D(
                source_id=source.id,
                target_id=target.id,
                edge_type=edge_type,
                weight=weight,
                distance=distance,
                elevation_change=target.elevation - source.elevation,
                properties={
                    'accessibility_friendly': edge_type == 'elevator',
                    'physical_effort': 'high' if edge_type == 'stairs' else 'low'
                }
            )
            
            self.edges_3d[edge_id] = edge
    
    def _add_horizontal_connections(self, node1: Node3D, node2: Node3D, distance: float):
        """Add horizontal connections between locations at same or adjacent elevations."""
        elevation_diff = abs(node1.elevation - node2.elevation)
        
        # Base walking speed: 1.4 m/s (5 km/h)
        walking_speed = 1.4
        base_time = distance / walking_speed
        
        if elevation_diff == 0:
            # Same elevation - direct walking
            edge_type = 'walk'
            weight = base_time
        else:
            # Adjacent elevations - might be ramp or gradual incline
            edge_type = 'ramp'
            weight = base_time * 1.3  # 30% penalty for incline
        
        # Add bidirectional edges
        for source, target in [(node1, node2), (node2, node1)]:
            edge_id = (source.id, target.id)
            
            edge = Edge3D(
                source_id=source.id,
                target_id=target.id,
                edge_type=edge_type,
                weight=weight,
                distance=distance,
                elevation_change=target.elevation - source.elevation,
                properties={
                    'accessibility_friendly': True,
                    'physical_effort': 'low' if elevation_diff == 0 else 'medium'
                }
            )
            
            self.edges_3d[edge_id] = edge
    
    def _build_networkx_graph(self):
        """Build NetworkX graph from 3D nodes and edges."""
        self.graph.clear()
        
        # Add nodes
        for node in self.nodes_3d.values():
            self.graph.add_node(
                node.id,
                lat=node.lat,
                lon=node.lon,
                elevation=node.elevation,
                node_type=node.node_type,
                **node.properties
            )
        
        # Add edges
        for edge in self.edges_3d.values():
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                weight=edge.weight,
                edge_type=edge.edge_type,
                distance=edge.distance,
                elevation_change=edge.elevation_change,
                **edge.properties
            )
    
    def add_custom_node(self, lat: float, lon: float, elevation: int, 
                       node_type: str = 'custom', properties: Dict = None) -> str:
        """Add a custom node to the graph."""
        node_id = f"custom_{lat}_{lon}_e{elevation}"
        
        node = Node3D(
            id=node_id,
            lat=lat,
            lon=lon,
            elevation=elevation,
            node_type=node_type,
            properties=properties or {}
        )
        
        self.nodes_3d[node_id] = node
        
        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            lat=node.lat,
            lon=node.lon,
            elevation=node.elevation,
            node_type=node.node_type,
            **node.properties
        )
        
        # Connect to nearby nodes
        self._connect_node_to_graph(node)
        
        return node_id
    
    def _connect_node_to_graph(self, new_node: Node3D):
        """Connect a new node to existing nodes in the graph."""
        for existing_node in self.nodes_3d.values():
            if existing_node.id == new_node.id:
                continue
                
            distance = geodesic((new_node.lat, new_node.lon), 
                              (existing_node.lat, existing_node.lon)).meters
            
            if distance <= 200:  # Within walking distance
                elevation_diff = abs(new_node.elevation - existing_node.elevation)
                
                if distance < 10 and elevation_diff > 0:
                    self._add_vertical_connections(new_node, existing_node, distance)
                elif elevation_diff <= 1:
                    self._add_horizontal_connections(new_node, existing_node, distance)
    
    def find_nodes_near_location(self, lat: float, lon: float, 
                                radius: float = 100) -> List[Node3D]:
        """Find all nodes within radius of given location."""
        nearby_nodes = []
        
        for node in self.nodes_3d.values():
            distance = geodesic((lat, lon), (node.lat, node.lon)).meters
            if distance <= radius:
                nearby_nodes.append(node)
        
        return sorted(nearby_nodes, key=lambda n: geodesic((lat, lon), (n.lat, n.lon)).meters)
    
    def get_elevation_levels_at_location(self, lat: float, lon: float, 
                                       radius: float = 50) -> List[int]:
        """Get all available elevation levels near a location."""
        nearby_nodes = self.find_nodes_near_location(lat, lon, radius)
        elevations = sorted(set(node.elevation for node in nearby_nodes))
        return elevations
    
    def visualize_graph_3d(self, figsize: Tuple[int, int] = (12, 8)):
        """Create 3D visualization of the graph."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        for node in self.nodes_3d.values():
            color = {'entrance': 'green', 'junction': 'red', 'location': 'blue', 'custom': 'purple'}.get(node.node_type, 'gray')
            ax.scatter(node.lon, node.lat, node.elevation, c=color, s=50, alpha=0.7)
        
        # Plot edges
        for edge in self.edges_3d.values():
            source_node = self.nodes_3d[edge.source_id]
            target_node = self.nodes_3d[edge.target_id]
            
            edge_color = {'walk': 'blue', 'stairs': 'red', 'elevator': 'green', 'ramp': 'orange'}.get(edge.edge_type, 'gray')
            
            ax.plot([source_node.lon, target_node.lon], 
                   [source_node.lat, target_node.lat], 
                   [source_node.elevation, target_node.elevation], 
                   color=edge_color, alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation Level')
        ax.set_title('3D Navigation Graph')
        
        # Add legend
        import matplotlib.patches as mpatches
        node_patches = [
            mpatches.Patch(color='green', label='Entrance'),
            mpatches.Patch(color='red', label='Junction'),
            mpatches.Patch(color='blue', label='Location'),
            mpatches.Patch(color='purple', label='Custom')
        ]
        edge_patches = [
            mpatches.Patch(color='blue', label='Walk'),
            mpatches.Patch(color='red', label='Stairs'),
            mpatches.Patch(color='green', label='Elevator'),
            mpatches.Patch(color='orange', label='Ramp')
        ]
        
        ax.legend(handles=node_patches + edge_patches, loc='upper left', bbox_to_anchor=(0, 1))
        
        return fig, ax
    
    def save_graph(self, filepath: str):
        """Save graph to file."""
        graph_data = {
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes_3d.items()},
            'edges': {f"{edge_key[0]}-{edge_key[1]}": asdict(edge) 
                     for edge_key, edge in self.edges_3d.items()},
            'networkx_graph': nx.node_link_data(self.graph)
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
    
    def load_graph(self, filepath: str):
        """Load graph from file."""
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Reconstruct nodes
        self.nodes_3d = {}
        for node_id, node_data in graph_data['nodes'].items():
            node = Node3D(**node_data)
            self.nodes_3d[node_id] = node
        
        # Reconstruct edges
        self.edges_3d = {}
        for edge_key, edge_data in graph_data['edges'].items():
            edge = Edge3D(**edge_data)
            source_id, target_id = edge_key.split('-', 1)
            self.edges_3d[(source_id, target_id)] = edge
        
        # Reconstruct NetworkX graph
        self.graph = nx.node_link_graph(graph_data['networkx_graph'], directed=True)


if __name__ == "__main__":
    # Example usage
    print("Testing 3D Graph Builder...")
    
    # Create synthetic data for testing
    from elevation_predictor import ElevationPredictor
    
    # Initialize components
    predictor = ElevationPredictor()
    features, labels = predictor.create_synthetic_training_data(n_samples=1000)
    predictor.train_model(features, labels)
    
    data_collector = DataCollector("data/test_sensor_data.db")
    
    # Create test data
    test_locations = [
        (29.5630, 106.5516, 0),  # Ground level
        (29.5631, 106.5517, 1),  # First floor
        (29.5632, 106.5518, 2),  # Second floor
        (29.5635, 106.5520, 0),  # Another ground level location
        (29.5636, 106.5521, 1),  # Another first floor
    ]
    
    import asyncio
    
    async def create_test_data():
        for lat, lon, elevation in test_locations:
            for _ in range(20):  # More samples per location
                sensor_reading = data_collector.simulate_sensor_reading(lat, lon, elevation)
                await data_collector.collect_sensor_data(sensor_reading)
    
    asyncio.run(create_test_data())
    
    # Build graph
    graph_builder = Graph3DBuilder(predictor)
    graph = graph_builder.build_graph_from_data(data_collector, cluster_radius=30)
    
    # Save graph
    graph_builder.save_graph("data/test_3d_graph.json")
    
    # Visualize graph
    try:
        fig, ax = graph_builder.visualize_graph_3d()
        plt.savefig("data/3d_graph_visualization.png", dpi=300, bbox_inches='tight')
        print("3D graph visualization saved")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print(f"\nGraph statistics:")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_weakly_connected_components(graph)}")
    
    # Test location queries
    test_lat, test_lon = 29.5630, 106.5516
    nearby_nodes = graph_builder.find_nodes_near_location(test_lat, test_lon, radius=100)
    print(f"\nNearby nodes to ({test_lat}, {test_lon}):")
    for node in nearby_nodes[:3]:
        distance = geodesic((test_lat, test_lon), (node.lat, node.lon)).meters
        print(f"  {node.id}: Elevation {node.elevation}, Distance {distance:.1f}m")