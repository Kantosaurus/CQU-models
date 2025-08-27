"""
Main entry point for the 3D Navigation System
Demonstrates the complete pipeline from data collection to navigation.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import DataCollector, SensorData
from elevation_predictor import ElevationPredictor
from graph_builder import Graph3DBuilder
from pathfinder import Dijkstra3D, RouteConstraints
from navigation_app import NavigationApp


async def demo_system():
    """Demonstrate the complete 3D navigation system."""
    print("🏢 3D Navigation System Demo")
    print("=" * 50)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("1. Initializing data collection...")
    data_collector = DataCollector("data/demo_sensor_data.db")
    
    # Simulate sensor data collection for Chongqing locations
    print("2. Simulating sensor data collection...")
    chongqing_locations = [
        # Jiefangbei CBD area (multi-level shopping and business district)
        (29.5630, 106.5516, 0), (29.5630, 106.5516, 1), (29.5630, 106.5516, 2),
        (29.5631, 106.5517, 0), (29.5631, 106.5517, 1), (29.5631, 106.5517, 2),
        (29.5632, 106.5518, 0), (29.5632, 106.5518, 1),
        
        # Nearby connected areas
        (29.5640, 106.5530, 0), (29.5640, 106.5530, 1),
        (29.5641, 106.5531, 0), (29.5641, 106.5531, 1), (29.5641, 106.5531, 2),
        
        # More distant but connected locations
        (29.5650, 106.5540, 0), (29.5650, 106.5540, 1),
        (29.5651, 106.5541, 0), (29.5651, 106.5541, 1), (29.5651, 106.5541, 2),
        (29.5652, 106.5542, 0), (29.5652, 106.5542, 1),
        
        # Bridge/tunnel connections
        (29.5660, 106.5550, 0), (29.5660, 106.5550, 1),
        (29.5670, 106.5560, 0), (29.5670, 106.5560, 1), (29.5670, 106.5560, 2),
    ]
    
    total_readings = 0
    for lat, lon, elevation in chongqing_locations:
        for _ in range(20):  # 20 readings per location for good coverage
            sensor_reading = data_collector.simulate_sensor_reading(lat, lon, elevation)
            await data_collector.collect_sensor_data(sensor_reading)
            total_readings += 1
    
    print(f"   ✓ Collected {total_readings} sensor readings from {len(chongqing_locations)} locations")
    
    print("3. Training elevation prediction model...")
    elevation_predictor = ElevationPredictor()
    
    # Train model with synthetic data
    features, labels = elevation_predictor.create_synthetic_training_data(n_samples=2000)
    metrics = elevation_predictor.train_model(features, labels)
    
    print(f"   ✓ Model accuracy: {metrics['accuracy']:.3f}")
    print(f"   ✓ Cross-validation accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
    
    # Save model
    elevation_predictor.save_model("models/elevation_predictor.joblib")
    print("   ✓ Model saved")
    
    print("4. Building 3D navigation graph...")
    graph_builder = Graph3DBuilder(elevation_predictor)
    
    # Build graph from collected data
    graph = graph_builder.build_graph_from_data(data_collector, cluster_radius=50, min_cluster_size=3)
    print(f"   ✓ Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Save graph
    graph_builder.save_graph("data/navigation_graph.json")
    print("   ✓ Graph saved")
    
    print("5. Testing pathfinding...")
    pathfinder = Dijkstra3D(graph_builder)
    
    # Test basic pathfinding
    start_lat, start_lon = 29.5630, 106.5516  # Jiefangbei area
    end_lat, end_lon = 29.5670, 106.5560      # Further location
    
    print(f"   Finding path from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
    
    # Basic route
    route = pathfinder.find_shortest_path(start_lat, start_lon, end_lat, end_lon)
    
    if route:
        summary = pathfinder.get_route_summary(route)
        print(f"   ✓ Basic route found:")
        print(f"     - Distance: {summary['total_distance_m']}m")
        print(f"     - Duration: {summary['total_duration_min']} minutes")
        print(f"     - Elevation change: {summary['elevation_change']} levels")
        print(f"     - Steps: {summary['total_steps']}")
        print(f"     - Quality score: {summary['quality_score']}")
        
        print("   ✓ Route steps:")
        for i, step in enumerate(route.steps[:5], 1):  # Show first 5 steps
            print(f"     {i}. {step.instruction}")
        if len(route.steps) > 5:
            print(f"     ... and {len(route.steps) - 5} more steps")
            
        # Export route
        pathfinder.export_route_gpx(route, "data/demo_route.gpx")
        print("   ✓ Route exported to GPX")
    else:
        print("   ⚠ No route found")
    
    # Test accessibility-friendly route
    print("6. Testing accessibility features...")
    accessible_constraints = RouteConstraints(
        avoid_stairs=True,
        accessibility_required=True
    )
    
    accessible_route = pathfinder.find_shortest_path(
        start_lat, start_lon, end_lat, end_lon,
        constraints=accessible_constraints
    )
    
    if accessible_route:
        acc_summary = pathfinder.get_route_summary(accessible_route)
        print(f"   ✓ Accessible route found:")
        print(f"     - Duration: {acc_summary['total_duration_min']} minutes")
        print(f"     - No stairs used: {'stairs' not in acc_summary['step_types']}")
    else:
        print("   ⚠ No accessible route found")
    
    # Test multiple routes
    print("7. Finding alternative routes...")
    alternatives = pathfinder.find_multiple_routes(
        start_lat, start_lon, end_lat, end_lon, num_routes=3
    )
    
    print(f"   ✓ Found {len(alternatives)} alternative routes:")
    for i, alt_route in enumerate(alternatives, 1):
        if alt_route:
            alt_summary = pathfinder.get_route_summary(alt_route)
            print(f"     Route {i}: {alt_summary['total_duration_min']} min, "
                  f"{alt_summary['total_distance_m']}m, "
                  f"Quality: {alt_summary['quality_score']}")
    
    print("8. System demonstration complete!")
    print("\n" + "=" * 50)
    print("🚀 Starting web application...")
    print("   Access the navigation system at: http://127.0.0.1:8000")
    print("   API documentation at: http://127.0.0.1:8000/docs")
    print("\nFeatures available:")
    print("   • Interactive map with navigation nodes")
    print("   • Real-time route planning with multiple alternatives")
    print("   • Accessibility-friendly route options")
    print("   • 3D elevation-aware pathfinding")
    print("   • Sensor data collection simulation")
    print("   • Dynamic graph rebuilding")
    
    return True


def run_web_app():
    """Run the web application."""
    app = NavigationApp(data_path="data")
    app.run(host="127.0.0.1", port=8000)


def main():
    """Main function - choose between demo or web app."""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demonstration
        success = asyncio.run(demo_system())
        if success:
            input("\nPress Enter to start the web application...")
            run_web_app()
    else:
        # Run web app directly
        print("🏢 3D Navigation System")
        print("Starting web application...")
        print("Run 'python main.py demo' to see the full system demonstration first.")
        run_web_app()


if __name__ == "__main__":
    main()