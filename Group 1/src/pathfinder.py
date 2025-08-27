"""
3D Dijkstra Pathfinding Algorithm
Computes optimal paths in 3D space with elevation constraints and preferences.
"""

import heapq
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from geopy.distance import geodesic
import json
from datetime import datetime, timedelta

from graph_builder import Graph3DBuilder, Node3D, Edge3D


@dataclass
class RouteConstraints:
    """Constraints for route planning."""
    avoid_stairs: bool = False
    avoid_elevators: bool = False
    prefer_escalators: bool = False
    max_elevation_change: int = 10
    accessibility_required: bool = False
    max_walking_distance: float = 1000  # meters
    max_total_time: float = 1800  # seconds (30 minutes)


@dataclass
class RouteStep:
    """Individual step in a route."""
    from_node: str
    to_node: str
    instruction: str
    distance: float
    duration: float
    elevation_change: int
    step_type: str  # 'walk', 'stairs', 'elevator', 'escalator'
    coordinates: Tuple[float, float, int]  # (lat, lon, elevation)


@dataclass
class Route:
    """Complete route with steps and metadata."""
    steps: List[RouteStep]
    total_distance: float
    total_duration: float
    total_elevation_change: int
    start_coordinates: Tuple[float, float, int]
    end_coordinates: Tuple[float, float, int]
    route_quality: float  # 0-1 score based on constraints satisfaction
    alternative_count: int = 0


class Dijkstra3D:
    """3D Dijkstra pathfinding implementation for elevation-aware navigation."""
    
    def __init__(self, graph_builder: Graph3DBuilder):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        self.nodes_3d = graph_builder.nodes_3d
        self.edges_3d = graph_builder.edges_3d
    
    def find_shortest_path(self, start_lat: float, start_lon: float, 
                          end_lat: float, end_lon: float,
                          constraints: Optional[RouteConstraints] = None,
                          start_elevation: Optional[int] = None,
                          end_elevation: Optional[int] = None) -> Optional[Route]:
        """
        Find shortest path between two coordinates using 3D Dijkstra algorithm.
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude
            end_lon: Ending longitude
            constraints: Route planning constraints
            start_elevation: Preferred starting elevation (None for auto-detect)
            end_elevation: Preferred ending elevation (None for auto-detect)
            
        Returns:
            Route object with path details, or None if no path found
        """
        if constraints is None:
            constraints = RouteConstraints()
        
        # Find start and end nodes
        start_nodes = self._find_best_nodes(start_lat, start_lon, start_elevation)
        end_nodes = self._find_best_nodes(end_lat, end_lon, end_elevation)
        
        if not start_nodes or not end_nodes:
            return None
        
        best_route = None
        best_cost = float('inf')
        
        # Try multiple start-end combinations
        for start_node_id in start_nodes[:3]:  # Try top 3 start nodes
            for end_node_id in end_nodes[:3]:    # Try top 3 end nodes
                try:
                    route = self._dijkstra_search(start_node_id, end_node_id, constraints)
                    if route and route.total_duration < best_cost:
                        best_route = route
                        best_cost = route.total_duration
                except Exception as e:
                    continue
        
        return best_route
    
    def find_multiple_routes(self, start_lat: float, start_lon: float,
                           end_lat: float, end_lon: float,
                           num_routes: int = 3,
                           constraints: Optional[RouteConstraints] = None) -> List[Route]:
        """Find multiple alternative routes between two points."""
        if constraints is None:
            constraints = RouteConstraints()
        
        routes = []
        
        # Find primary route
        primary_route = self.find_shortest_path(
            start_lat, start_lon, end_lat, end_lon, constraints
        )
        
        if primary_route:
            routes.append(primary_route)
        
        # Generate alternatives with modified constraints
        constraint_variations = [
            RouteConstraints(**{**constraints.__dict__, 'avoid_stairs': True}),
            RouteConstraints(**{**constraints.__dict__, 'prefer_escalators': True}),
            RouteConstraints(**{**constraints.__dict__, 'accessibility_required': True}),
        ]
        
        for alt_constraints in constraint_variations:
            if len(routes) >= num_routes:
                break
                
            alt_route = self.find_shortest_path(
                start_lat, start_lon, end_lat, end_lon, alt_constraints
            )
            
            if alt_route and not self._routes_too_similar(alt_route, routes):
                alt_route.alternative_count = len(routes)
                routes.append(alt_route)
        
        return routes
    
    def _find_best_nodes(self, lat: float, lon: float, 
                        preferred_elevation: Optional[int] = None) -> List[str]:
        """Find best nodes near a coordinate, optionally preferring an elevation."""
        nearby_nodes = self.graph_builder.find_nodes_near_location(lat, lon, radius=100)
        
        if not nearby_nodes:
            # Expand search radius
            nearby_nodes = self.graph_builder.find_nodes_near_location(lat, lon, radius=200)
        
        if not nearby_nodes:
            return []
        
        # Score nodes based on distance and elevation preference
        node_scores = []
        for node in nearby_nodes:
            distance = geodesic((lat, lon), (node.lat, node.lon)).meters
            distance_score = 1.0 / (1.0 + distance / 100.0)  # Prefer closer nodes
            
            elevation_score = 1.0
            if preferred_elevation is not None:
                elevation_diff = abs(node.elevation - preferred_elevation)
                elevation_score = 1.0 / (1.0 + elevation_diff)  # Prefer matching elevation
            
            # Prefer entrance nodes for better accessibility
            type_score = 1.2 if node.node_type == 'entrance' else 1.0
            
            total_score = distance_score * elevation_score * type_score
            node_scores.append((node.id, total_score))
        
        # Sort by score and return node IDs
        node_scores.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in node_scores]
    
    def _dijkstra_search(self, start_node_id: str, end_node_id: str, 
                        constraints: RouteConstraints) -> Optional[Route]:
        """Core Dijkstra algorithm implementation with 3D constraints."""
        # Priority queue: (cost, node_id, path, total_distance, elevation_changes)
        pq = [(0, start_node_id, [start_node_id], 0, 0)]
        visited = set()
        
        while pq:
            current_cost, current_node, path, total_distance, elevation_changes = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Check if we reached the destination
            if current_node == end_node_id:
                return self._build_route(path, constraints)
            
            # Check constraints
            if (total_distance > constraints.max_walking_distance or 
                current_cost > constraints.max_total_time or
                abs(elevation_changes) > constraints.max_elevation_change):
                continue
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                edge_cost = self._calculate_edge_cost(current_node, neighbor, edge_data, constraints)
                
                # Skip if edge violates constraints
                if edge_cost == float('inf'):
                    continue
                
                new_cost = current_cost + edge_cost
                new_distance = total_distance + edge_data.get('distance', 0)
                new_elevation_changes = elevation_changes + edge_data.get('elevation_change', 0)
                new_path = path + [neighbor]
                
                heapq.heappush(pq, (new_cost, neighbor, new_path, new_distance, new_elevation_changes))
        
        return None  # No path found
    
    def _calculate_edge_cost(self, from_node: str, to_node: str, 
                           edge_data: Dict, constraints: RouteConstraints) -> float:
        """Calculate the cost of traversing an edge based on constraints."""
        base_cost = edge_data.get('weight', 60)  # Default 1 minute
        edge_type = edge_data.get('edge_type', 'walk')
        
        # Apply constraint filters
        if constraints.avoid_stairs and edge_type == 'stairs':
            return float('inf')
        
        if constraints.avoid_elevators and edge_type == 'elevator':
            return float('inf')
        
        if constraints.accessibility_required and not edge_data.get('accessibility_friendly', False):
            return float('inf')
        
        # Apply preferences
        cost_multiplier = 1.0
        
        if constraints.prefer_escalators and edge_type == 'escalator':
            cost_multiplier *= 0.8  # 20% preference bonus
        
        # Penalize high physical effort if accessibility is a concern
        physical_effort = edge_data.get('physical_effort', 'low')
        if constraints.accessibility_required and physical_effort == 'high':
            cost_multiplier *= 1.5
        
        # Penalize large elevation changes
        elevation_change = abs(edge_data.get('elevation_change', 0))
        if elevation_change > 1:
            cost_multiplier *= (1.0 + 0.2 * elevation_change)
        
        return base_cost * cost_multiplier
    
    def _build_route(self, path: List[str], constraints: RouteConstraints) -> Route:
        """Build Route object from node path."""
        steps = []
        total_distance = 0
        total_duration = 0
        total_elevation_change = 0
        
        for i in range(len(path) - 1):
            from_node_id = path[i]
            to_node_id = path[i + 1]
            
            from_node = self.nodes_3d[from_node_id]
            to_node = self.nodes_3d[to_node_id]
            
            # Get edge data
            edge_data = self.graph.get_edge_data(from_node_id, to_node_id)
            
            # Calculate step details
            distance = edge_data.get('distance', 0)
            duration = edge_data.get('weight', 60)
            elevation_change = edge_data.get('elevation_change', 0)
            step_type = edge_data.get('edge_type', 'walk')
            
            # Generate instruction
            instruction = self._generate_instruction(from_node, to_node, step_type, distance, elevation_change)
            
            step = RouteStep(
                from_node=from_node_id,
                to_node=to_node_id,
                instruction=instruction,
                distance=distance,
                duration=duration,
                elevation_change=elevation_change,
                step_type=step_type,
                coordinates=(to_node.lat, to_node.lon, to_node.elevation)
            )
            
            steps.append(step)
            total_distance += distance
            total_duration += duration
            total_elevation_change += elevation_change
        
        # Calculate route quality score
        quality = self._calculate_route_quality(steps, constraints)
        
        start_node = self.nodes_3d[path[0]]
        end_node = self.nodes_3d[path[-1]]
        
        return Route(
            steps=steps,
            total_distance=total_distance,
            total_duration=total_duration,
            total_elevation_change=total_elevation_change,
            start_coordinates=(start_node.lat, start_node.lon, start_node.elevation),
            end_coordinates=(end_node.lat, end_node.lon, end_node.elevation),
            route_quality=quality
        )
    
    def _generate_instruction(self, from_node: Node3D, to_node: Node3D, 
                            step_type: str, distance: float, elevation_change: int) -> str:
        """Generate human-readable navigation instruction."""
        if step_type == 'walk':
            if distance < 50:
                return f"Walk {distance:.0f}m to next point"
            else:
                return f"Walk {distance:.0f}m along the path"
        
        elif step_type == 'stairs':
            if elevation_change > 0:
                return f"Take stairs up {elevation_change} level(s)"
            else:
                return f"Take stairs down {abs(elevation_change)} level(s)"
        
        elif step_type == 'elevator':
            if elevation_change > 0:
                return f"Take elevator up to level {to_node.elevation}"
            else:
                return f"Take elevator down to level {to_node.elevation}"
        
        elif step_type == 'escalator':
            if elevation_change > 0:
                return f"Take escalator up {elevation_change} level(s)"
            else:
                return f"Take escalator down {abs(elevation_change)} level(s)"
        
        elif step_type == 'ramp':
            return f"Follow ramp for {distance:.0f}m"
        
        else:
            return f"Continue {distance:.0f}m"
    
    def _calculate_route_quality(self, steps: List[RouteStep], 
                               constraints: RouteConstraints) -> float:
        """Calculate route quality score based on how well it satisfies constraints."""
        score = 1.0
        
        # Penalize constraint violations
        for step in steps:
            if constraints.avoid_stairs and step.step_type == 'stairs':
                score *= 0.7
            if constraints.avoid_elevators and step.step_type == 'elevator':
                score *= 0.7
        
        # Reward preference satisfaction
        escalator_steps = sum(1 for step in steps if step.step_type == 'escalator')
        if constraints.prefer_escalators and escalator_steps > 0:
            score *= 1.2
        
        # Penalize excessive elevation changes
        total_elevation_change = sum(abs(step.elevation_change) for step in steps)
        if total_elevation_change > 3:
            score *= 0.9 ** (total_elevation_change - 3)
        
        return max(0.0, min(1.0, score))
    
    def _routes_too_similar(self, new_route: Route, existing_routes: List[Route]) -> bool:
        """Check if a new route is too similar to existing ones."""
        for existing in existing_routes:
            # Compare route steps
            if len(new_route.steps) == len(existing.steps):
                same_steps = sum(1 for i in range(len(new_route.steps))
                               if new_route.steps[i].to_node == existing.steps[i].to_node)
                similarity = same_steps / len(new_route.steps)
                
                if similarity > 0.8:  # 80% similarity threshold
                    return True
        
        return False
    
    def get_route_summary(self, route: Route) -> Dict:
        """Get a summary of route information."""
        step_types = {}
        for step in route.steps:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
        
        return {
            'total_steps': len(route.steps),
            'total_distance_m': round(route.total_distance, 1),
            'total_duration_min': round(route.total_duration / 60, 1),
            'elevation_change': route.total_elevation_change,
            'step_types': step_types,
            'quality_score': round(route.route_quality, 2),
            'start_elevation': route.start_coordinates[2],
            'end_elevation': route.end_coordinates[2]
        }
    
    def export_route_gpx(self, route: Route, filename: str):
        """Export route to GPX format for GPS devices."""
        gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        gpx_content += '<gpx version="1.1" creator="3D Navigation System">\n'
        gpx_content += '  <trk>\n'
        gpx_content += '    <name>3D Navigation Route</name>\n'
        gpx_content += '    <trkseg>\n'
        
        # Add start point
        start = route.start_coordinates
        gpx_content += f'    <trkpt lat="{start[0]}" lon="{start[1]}">\n'
        gpx_content += f'      <ele>{start[2] * 3}</ele>\n'  # Convert floor to meters
        gpx_content += '    </trkpt>\n'
        
        # Add waypoints for each step
        for step in route.steps:
            coords = step.coordinates
            gpx_content += f'    <trkpt lat="{coords[0]}" lon="{coords[1]}">\n'
            gpx_content += f'      <ele>{coords[2] * 3}</ele>\n'
            gpx_content += f'      <desc>{step.instruction}</desc>\n'
            gpx_content += '    </trkpt>\n'
        
        gpx_content += '    </trkseg>\n'
        gpx_content += '  </trk>\n'
        gpx_content += '</gpx>\n'
        
        with open(filename, 'w') as f:
            f.write(gpx_content)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing 3D Dijkstra Pathfinding...")
    
    # Set up test environment
    from elevation_predictor import ElevationPredictor
    from data_collector import DataCollector
    
    # Initialize components
    predictor = ElevationPredictor()
    features, labels = predictor.create_synthetic_training_data(n_samples=2000)
    predictor.train_model(features, labels)
    
    data_collector = DataCollector("data/pathfinder_test.db")
    
    # Create comprehensive test data
    test_locations = [
        (29.5630, 106.5516, 0),  # Point A - Ground
        (29.5631, 106.5517, 1),  # Point A - Floor 1
        (29.5632, 106.5518, 2),  # Point A - Floor 2
        (29.5640, 106.5530, 0),  # Point B - Ground
        (29.5641, 106.5531, 1),  # Point B - Floor 1
        (29.5650, 106.5540, 0),  # Point C - Ground
        (29.5651, 106.5541, 1),  # Point C - Floor 1
        (29.5652, 106.5542, 2),  # Point C - Floor 2
    ]
    
    import asyncio
    
    async def create_test_data():
        for lat, lon, elevation in test_locations:
            for _ in range(25):
                sensor_reading = data_collector.simulate_sensor_reading(lat, lon, elevation)
                await data_collector.collect_sensor_data(sensor_reading)
    
    asyncio.run(create_test_data())
    
    # Build graph
    graph_builder = Graph3DBuilder(predictor)
    graph = graph_builder.build_graph_from_data(data_collector, cluster_radius=50)
    
    # Initialize pathfinder
    pathfinder = Dijkstra3D(graph_builder)
    
    # Test pathfinding
    start_lat, start_lon = 29.5630, 106.5516
    end_lat, end_lon = 29.5650, 106.5540
    
    print(f"\nFinding path from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
    
    # Test 1: Basic pathfinding
    route = pathfinder.find_shortest_path(start_lat, start_lon, end_lat, end_lon)
    
    if route:
        summary = pathfinder.get_route_summary(route)
        print(f"\nBasic Route Found:")
        print(f"  Distance: {summary['total_distance_m']}m")
        print(f"  Duration: {summary['total_duration_min']} minutes")
        print(f"  Elevation change: {summary['elevation_change']}")
        print(f"  Quality: {summary['quality_score']}")
        print(f"  Step types: {summary['step_types']}")
        
        print("\nRoute Steps:")
        for i, step in enumerate(route.steps, 1):
            print(f"  {i}. {step.instruction}")
    else:
        print("No route found!")
    
    # Test 2: Accessibility-friendly route
    accessible_constraints = RouteConstraints(
        avoid_stairs=True,
        accessibility_required=True
    )
    
    accessible_route = pathfinder.find_shortest_path(
        start_lat, start_lon, end_lat, end_lon, 
        constraints=accessible_constraints
    )
    
    if accessible_route:
        summary = pathfinder.get_route_summary(accessible_route)
        print(f"\nAccessible Route Found:")
        print(f"  Distance: {summary['total_distance_m']}m")
        print(f"  Duration: {summary['total_duration_min']} minutes")
        print(f"  Quality: {summary['quality_score']}")
    
    # Test 3: Multiple alternative routes
    alternatives = pathfinder.find_multiple_routes(
        start_lat, start_lon, end_lat, end_lon, num_routes=3
    )
    
    print(f"\nFound {len(alternatives)} alternative routes:")
    for i, alt_route in enumerate(alternatives, 1):
        summary = pathfinder.get_route_summary(alt_route)
        print(f"  Route {i}: {summary['total_duration_min']} min, "
              f"{summary['total_distance_m']}m, Quality: {summary['quality_score']}")
    
    # Export route to GPX
    if route:
        pathfinder.export_route_gpx(route, "data/test_route.gpx")
        print("\nRoute exported to GPX format")
    
    print("\nPathfinding test completed!")