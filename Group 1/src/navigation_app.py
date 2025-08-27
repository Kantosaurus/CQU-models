"""
Navigation App Interface
Provides user interface and real-time navigation output for the 3D navigation system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import json
import folium
from folium.plugins import MarkerCluster
import webbrowser
from typing import List, Dict, Optional, Tuple
import uvicorn
from datetime import datetime

from data_collector import DataCollector, SensorData
from elevation_predictor import ElevationPredictor, ElevationTracker
from graph_builder import Graph3DBuilder
from pathfinder import Dijkstra3D, RouteConstraints, Route


class NavigationRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    avoid_stairs: bool = False
    avoid_elevators: bool = False
    prefer_escalators: bool = False
    accessibility_required: bool = False
    num_alternatives: int = 1


class SensorDataRequest(BaseModel):
    user_id: str
    gps_lat: float
    gps_lon: float
    gps_accuracy: float
    wifi_signals: List[Dict[str, float]]
    bluetooth_beacons: List[Dict[str, float]]
    barometer_pressure: float
    accelerometer_x: float
    accelerometer_y: float
    accelerometer_z: float
    gyroscope_x: float
    gyroscope_y: float
    gyroscope_z: float
    magnetometer_x: float
    magnetometer_y: float
    magnetometer_z: float


class NavigationApp:
    """Main navigation application with web interface."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.data_collector = DataCollector(f"{data_path}/navigation_data.db")
        self.elevation_predictor = ElevationPredictor()
        self.graph_builder = None
        self.pathfinder = None
        self.elevation_tracker = ElevationTracker()
        self.app = FastAPI(title="3D Navigation System", version="1.0.0")
        
        # Initialize system
        self._setup_routes()
        self._initialize_ml_models()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            return self._generate_web_interface()
        
        @self.app.post("/api/submit_sensor_data")
        async def submit_sensor_data(sensor_request: SensorDataRequest):
            """Accept sensor data from mobile devices."""
            try:
                sensor_data = SensorData(
                    timestamp=datetime.now(),
                    user_id=sensor_request.user_id,
                    gps_lat=sensor_request.gps_lat,
                    gps_lon=sensor_request.gps_lon,
                    gps_accuracy=sensor_request.gps_accuracy,
                    wifi_signals=sensor_request.wifi_signals,
                    bluetooth_beacons=sensor_request.bluetooth_beacons,
                    barometer_pressure=sensor_request.barometer_pressure,
                    accelerometer_x=sensor_request.accelerometer_x,
                    accelerometer_y=sensor_request.accelerometer_y,
                    accelerometer_z=sensor_request.accelerometer_z,
                    gyroscope_x=sensor_request.gyroscope_x,
                    gyroscope_y=sensor_request.gyroscope_y,
                    gyroscope_z=sensor_request.gyroscope_z,
                    magnetometer_x=sensor_request.magnetometer_x,
                    magnetometer_y=sensor_request.magnetometer_y,
                    magnetometer_z=sensor_request.magnetometer_z
                )
                
                success = await self.data_collector.collect_sensor_data(sensor_data)
                
                if success:
                    # Predict elevation for immediate feedback
                    from elevation_predictor import DataPreprocessor
                    features = DataPreprocessor.extract_features(sensor_data.__dict__)
                    elevation, confidence = self.elevation_predictor.predict_elevation(features)
                    
                    # Update elevation tracker
                    self.elevation_tracker.add_prediction(elevation, confidence)
                    smoothed_elevation, avg_confidence = self.elevation_tracker.get_smoothed_elevation()
                    
                    return {
                        "status": "success",
                        "predicted_elevation": elevation,
                        "confidence": confidence,
                        "smoothed_elevation": smoothed_elevation,
                        "smoothed_confidence": avg_confidence
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to store sensor data")
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/find_route")
        async def find_route(nav_request: NavigationRequest):
            """Find navigation route between two points."""
            try:
                if not self.pathfinder:
                    raise HTTPException(status_code=503, detail="Navigation system not initialized")
                
                constraints = RouteConstraints(
                    avoid_stairs=nav_request.avoid_stairs,
                    avoid_elevators=nav_request.avoid_elevators,
                    prefer_escalators=nav_request.prefer_escalators,
                    accessibility_required=nav_request.accessibility_required
                )
                
                if nav_request.num_alternatives > 1:
                    routes = self.pathfinder.find_multiple_routes(
                        nav_request.start_lat, nav_request.start_lon,
                        nav_request.end_lat, nav_request.end_lon,
                        num_routes=nav_request.num_alternatives,
                        constraints=constraints
                    )
                else:
                    route = self.pathfinder.find_shortest_path(
                        nav_request.start_lat, nav_request.start_lon,
                        nav_request.end_lat, nav_request.end_lon,
                        constraints=constraints
                    )
                    routes = [route] if route else []
                
                if not routes:
                    return {"status": "no_route", "routes": []}
                
                # Convert routes to JSON format
                routes_json = []
                for route in routes:
                    if route:
                        route_data = {
                            "steps": [
                                {
                                    "instruction": step.instruction,
                                    "distance": step.distance,
                                    "duration": step.duration,
                                    "elevation_change": step.elevation_change,
                                    "step_type": step.step_type,
                                    "coordinates": step.coordinates
                                }
                                for step in route.steps
                            ],
                            "total_distance": route.total_distance,
                            "total_duration": route.total_duration,
                            "total_elevation_change": route.total_elevation_change,
                            "route_quality": route.route_quality,
                            "summary": self.pathfinder.get_route_summary(route)
                        }
                        routes_json.append(route_data)
                
                return {
                    "status": "success",
                    "routes": routes_json,
                    "count": len(routes_json)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/elevation_levels")
        async def get_elevation_levels(lat: float, lon: float, radius: float = 50):
            """Get available elevation levels near a location."""
            try:
                if not self.graph_builder:
                    return {"elevations": []}
                
                elevations = self.graph_builder.get_elevation_levels_at_location(lat, lon, radius)
                return {"elevations": elevations}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/map_data")
        async def get_map_data():
            """Get map visualization data."""
            try:
                if not self.graph_builder:
                    return {"nodes": [], "edges": []}
                
                nodes_data = []
                for node in self.graph_builder.nodes_3d.values():
                    nodes_data.append({
                        "id": node.id,
                        "lat": node.lat,
                        "lon": node.lon,
                        "elevation": node.elevation,
                        "type": node.node_type,
                        "properties": node.properties
                    })
                
                edges_data = []
                for edge in self.graph_builder.edges_3d.values():
                    source_node = self.graph_builder.nodes_3d[edge.source_id]
                    target_node = self.graph_builder.nodes_3d[edge.target_id]
                    edges_data.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": edge.edge_type,
                        "weight": edge.weight,
                        "distance": edge.distance,
                        "source_coords": [source_node.lat, source_node.lon, source_node.elevation],
                        "target_coords": [target_node.lat, target_node.lon, target_node.elevation]
                    })
                
                return {
                    "nodes": nodes_data,
                    "edges": edges_data
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/rebuild_graph")
        async def rebuild_graph():
            """Rebuild the navigation graph with latest data."""
            try:
                await self._rebuild_graph()
                return {"status": "success", "message": "Graph rebuilt successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _initialize_ml_models(self):
        """Initialize machine learning models."""
        try:
            # Try to load existing model
            self.elevation_predictor.load_model(f"{self.data_path}/elevation_predictor.joblib")
            print("Loaded existing elevation prediction model")
        except:
            # Train new model with synthetic data
            print("Training new elevation prediction model...")
            features, labels = self.elevation_predictor.create_synthetic_training_data(n_samples=2000)
            metrics = self.elevation_predictor.train_model(features, labels)
            self.elevation_predictor.save_model(f"{self.data_path}/elevation_predictor.joblib")
            print(f"Model trained with accuracy: {metrics['accuracy']:.3f}")
        
        # Initialize graph builder
        asyncio.create_task(self._initialize_graph())
    
    async def _initialize_graph(self):
        """Initialize the navigation graph."""
        try:
            self.graph_builder = Graph3DBuilder(self.elevation_predictor)
            
            # Try to load existing graph
            try:
                self.graph_builder.load_graph(f"{self.data_path}/navigation_graph.json")
                print("Loaded existing navigation graph")
            except:
                # Build new graph from data
                print("Building new navigation graph...")
                await self._rebuild_graph()
            
            # Initialize pathfinder
            self.pathfinder = Dijkstra3D(self.graph_builder)
            print("Navigation system initialized successfully")
            
        except Exception as e:
            print(f"Error initializing graph: {e}")
    
    async def _rebuild_graph(self):
        """Rebuild the navigation graph from collected data."""
        if not self.graph_builder:
            self.graph_builder = Graph3DBuilder(self.elevation_predictor)
        
        # Build graph from collected data
        graph = self.graph_builder.build_graph_from_data(self.data_collector, cluster_radius=50)
        
        # Save graph
        self.graph_builder.save_graph(f"{self.data_path}/navigation_graph.json")
        
        # Update pathfinder
        self.pathfinder = Dijkstra3D(self.graph_builder)
        
        print(f"Graph rebuilt with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    def _generate_web_interface(self) -> str:
        """Generate the main web interface HTML."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>3D Navigation System</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    border-radius: 10px; 
                    padding: 20px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .header { 
                    text-align: center; 
                    margin-bottom: 30px; 
                    color: #333;
                }
                .controls { 
                    display: flex; 
                    gap: 20px; 
                    margin-bottom: 20px; 
                    flex-wrap: wrap;
                }
                .control-group { 
                    flex: 1; 
                    min-width: 200px; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px;
                    background: #fafafa;
                }
                .control-group h3 { 
                    margin-top: 0; 
                    color: #555;
                }
                .form-group { 
                    margin-bottom: 15px; 
                }
                .form-group label { 
                    display: block; 
                    margin-bottom: 5px; 
                    font-weight: bold;
                }
                .form-group input, .form-group select { 
                    width: 100%; 
                    padding: 8px; 
                    border: 1px solid #ccc; 
                    border-radius: 4px; 
                }
                .checkbox-group { 
                    display: flex; 
                    flex-direction: column; 
                    gap: 8px; 
                }
                .checkbox-group label { 
                    display: flex; 
                    align-items: center; 
                    font-weight: normal; 
                }
                .checkbox-group input { 
                    width: auto; 
                    margin-right: 8px; 
                }
                .button { 
                    background: #007bff; 
                    color: white; 
                    border: none; 
                    padding: 12px 24px; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    font-size: 16px;
                    margin: 5px;
                }
                .button:hover { 
                    background: #0056b3; 
                }
                .button.secondary { 
                    background: #6c757d; 
                }
                .button.secondary:hover { 
                    background: #545b62; 
                }
                #map { 
                    height: 400px; 
                    border-radius: 5px; 
                    border: 1px solid #ddd; 
                }
                .results { 
                    margin-top: 20px; 
                }
                .route { 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    margin-bottom: 15px; 
                    padding: 15px; 
                    background: #f9f9f9;
                }
                .route-header { 
                    font-weight: bold; 
                    margin-bottom: 10px; 
                    color: #333;
                }
                .route-summary { 
                    display: flex; 
                    gap: 20px; 
                    margin-bottom: 10px; 
                    font-size: 14px;
                }
                .route-steps { 
                    margin-top: 10px; 
                }
                .route-step { 
                    padding: 8px; 
                    border-left: 3px solid #007bff; 
                    margin-bottom: 5px; 
                    background: white; 
                    border-radius: 0 5px 5px 0;
                }
                .status { 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-bottom: 15px; 
                }
                .status.success { 
                    background: #d4edda; 
                    color: #155724; 
                    border: 1px solid #c3e6cb; 
                }
                .status.error { 
                    background: #f8d7da; 
                    color: #721c24; 
                    border: 1px solid #f5c6cb; 
                }
                .elevation-info {
                    font-size: 12px;
                    color: #666;
                    margin-top: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏢 3D Navigation System</h1>
                    <p>Navigate Chongqing's multi-level terrain with AI-powered 3D pathfinding</p>
                </div>
                
                <div class="controls">
                    <div class="control-group">
                        <h3>🎯 Navigation</h3>
                        <div class="form-group">
                            <label>Start Location (Lat, Lon)</label>
                            <input type="text" id="startLat" placeholder="29.5630" value="29.5630">
                            <input type="text" id="startLon" placeholder="106.5516" value="106.5516">
                        </div>
                        <div class="form-group">
                            <label>End Location (Lat, Lon)</label>
                            <input type="text" id="endLat" placeholder="29.5650" value="29.5650">
                            <input type="text" id="endLon" placeholder="106.5540" value="106.5540">
                        </div>
                        <div class="form-group">
                            <label>Number of Routes</label>
                            <select id="numRoutes">
                                <option value="1">1 Route</option>
                                <option value="2">2 Routes</option>
                                <option value="3" selected>3 Routes</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <h3>⚙️ Preferences</h3>
                        <div class="checkbox-group">
                            <label><input type="checkbox" id="avoidStairs"> Avoid Stairs</label>
                            <label><input type="checkbox" id="avoidElevators"> Avoid Elevators</label>
                            <label><input type="checkbox" id="preferEscalators"> Prefer Escalators</label>
                            <label><input type="checkbox" id="accessibilityRequired"> Accessibility Required</label>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <h3>🔧 System</h3>
                        <button class="button" onclick="findRoute()">Find Route</button>
                        <button class="button secondary" onclick="rebuildGraph()">Rebuild Graph</button>
                        <button class="button secondary" onclick="simulateData()">Simulate Data</button>
                    </div>
                </div>
                
                <div id="map"></div>
                
                <div id="status"></div>
                <div id="results" class="results"></div>
            </div>
            
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script>
                // Initialize map centered on Chongqing
                const map = L.map('map').setView([29.5630, 106.5516], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 18,
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                
                let startMarker = null;
                let endMarker = null;
                let routeLayer = null;
                let nodesLayer = null;
                
                // Map click handlers for setting start/end points
                let settingStart = false;
                let settingEnd = false;
                
                map.on('click', function(e) {
                    if (settingStart) {
                        if (startMarker) map.removeLayer(startMarker);
                        startMarker = L.marker([e.latlng.lat, e.latlng.lng])
                            .addTo(map)
                            .bindPopup('Start Location');
                        document.getElementById('startLat').value = e.latlng.lat.toFixed(6);
                        document.getElementById('startLon').value = e.latlng.lng.toFixed(6);
                        settingStart = false;
                    } else if (settingEnd) {
                        if (endMarker) map.removeLayer(endMarker);
                        endMarker = L.marker([e.latlng.lat, e.latlng.lng])
                            .addTo(map)
                            .bindPopup('End Location');
                        document.getElementById('endLat').value = e.latlng.lat.toFixed(6);
                        document.getElementById('endLon').value = e.latlng.lng.toFixed(6);
                        settingEnd = false;
                    }
                });
                
                function setStatus(message, type = 'success') {
                    const statusDiv = document.getElementById('status');
                    statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
                }
                
                async function findRoute() {
                    const startLat = parseFloat(document.getElementById('startLat').value);
                    const startLon = parseFloat(document.getElementById('startLon').value);
                    const endLat = parseFloat(document.getElementById('endLat').value);
                    const endLon = parseFloat(document.getElementById('endLon').value);
                    const numRoutes = parseInt(document.getElementById('numRoutes').value);
                    
                    const requestData = {
                        start_lat: startLat,
                        start_lon: startLon,
                        end_lat: endLat,
                        end_lon: endLon,
                        num_alternatives: numRoutes,
                        avoid_stairs: document.getElementById('avoidStairs').checked,
                        avoid_elevators: document.getElementById('avoidElevators').checked,
                        prefer_escalators: document.getElementById('preferEscalators').checked,
                        accessibility_required: document.getElementById('accessibilityRequired').checked
                    };
                    
                    setStatus('Finding route...', 'success');
                    
                    try {
                        const response = await fetch('/api/find_route', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(requestData)
                        });
                        
                        const result = await response.json();
                        
                        if (result.status === 'success' && result.routes.length > 0) {
                            displayRoutes(result.routes);
                            visualizeRoutesOnMap(result.routes);
                            setStatus(`Found ${result.count} route(s)`, 'success');
                        } else {
                            setStatus('No routes found. Try different locations or preferences.', 'error');
                            document.getElementById('results').innerHTML = '';
                        }
                        
                    } catch (error) {
                        setStatus(`Error finding route: ${error.message}`, 'error');
                        console.error('Route finding error:', error);
                    }
                }
                
                function displayRoutes(routes) {
                    const resultsDiv = document.getElementById('results');
                    let html = '<h2>🗺️ Navigation Routes</h2>';
                    
                    routes.forEach((route, index) => {
                        const summary = route.summary;
                        html += `
                            <div class="route">
                                <div class="route-header">
                                    Route ${index + 1} ${index === 0 ? '(Recommended)' : '(Alternative)'}
                                </div>
                                <div class="route-summary">
                                    <span>📏 ${summary.total_distance_m}m</span>
                                    <span>⏱️ ${summary.total_duration_min} min</span>
                                    <span>📈 ${summary.elevation_change > 0 ? '+' : ''}${summary.elevation_change} floors</span>
                                    <span>⭐ Quality: ${summary.quality_score}</span>
                                </div>
                                <div class="elevation-info">
                                    From Floor ${summary.start_elevation} to Floor ${summary.end_elevation}
                                </div>
                                <div class="route-steps">
                                    <strong>📋 Instructions:</strong>
                        `;
                        
                        route.steps.forEach((step, stepIndex) => {
                            const icon = getStepIcon(step.step_type);
                            html += `
                                <div class="route-step">
                                    ${stepIndex + 1}. ${icon} ${step.instruction}
                                    <div class="elevation-info">
                                        ${step.distance.toFixed(0)}m, ${(step.duration/60).toFixed(1)} min
                                        ${step.elevation_change !== 0 ? `, ${step.elevation_change > 0 ? '+' : ''}${step.elevation_change} floors` : ''}
                                    </div>
                                </div>
                            `;
                        });
                        
                        html += '</div></div>';
                    });
                    
                    resultsDiv.innerHTML = html;
                }
                
                function getStepIcon(stepType) {
                    const icons = {
                        'walk': '🚶',
                        'stairs': '🪜',
                        'elevator': '🛗',
                        'escalator': '🪃',
                        'ramp': '🛤️'
                    };
                    return icons[stepType] || '➡️';
                }
                
                function visualizeRoutesOnMap(routes) {
                    // Clear existing route visualization
                    if (routeLayer) {
                        map.removeLayer(routeLayer);
                    }
                    
                    routeLayer = L.layerGroup().addTo(map);
                    
                    routes.forEach((route, routeIndex) => {
                        const colors = ['#FF0000', '#00FF00', '#0000FF'];
                        const color = colors[routeIndex % colors.length];
                        
                        // Create route line
                        const routeCoords = [];
                        if (route.steps.length > 0) {
                            // Add start point
                            const firstStep = route.steps[0];
                            routeCoords.push([firstStep.coordinates[0], firstStep.coordinates[1]]);
                            
                            // Add all step endpoints
                            route.steps.forEach(step => {
                                routeCoords.push([step.coordinates[0], step.coordinates[1]]);
                            });
                        }
                        
                        if (routeCoords.length > 1) {
                            const polyline = L.polyline(routeCoords, {
                                color: color,
                                weight: 4,
                                opacity: 0.7
                            }).addTo(routeLayer);
                            
                            polyline.bindPopup(`Route ${routeIndex + 1}: ${route.summary.total_distance_m}m, ${route.summary.total_duration_min} min`);
                        }
                        
                        // Add markers for elevation changes
                        route.steps.forEach((step, stepIndex) => {
                            if (Math.abs(step.elevation_change) > 0) {
                                const marker = L.circleMarker([step.coordinates[0], step.coordinates[1]], {
                                    radius: 6,
                                    fillColor: step.elevation_change > 0 ? '#FF4444' : '#4444FF',
                                    color: '#000',
                                    weight: 1,
                                    opacity: 1,
                                    fillOpacity: 0.8
                                }).addTo(routeLayer);
                                
                                marker.bindPopup(`${step.instruction}<br>Floor change: ${step.elevation_change > 0 ? '+' : ''}${step.elevation_change}`);
                            }
                        });
                    });
                }
                
                async function rebuildGraph() {
                    setStatus('Rebuilding navigation graph...', 'success');
                    try {
                        const response = await fetch('/api/rebuild_graph', { method: 'POST' });
                        const result = await response.json();
                        setStatus(result.message, 'success');
                        loadMapData(); // Refresh map visualization
                    } catch (error) {
                        setStatus(`Error rebuilding graph: ${error.message}`, 'error');
                    }
                }
                
                async function simulateData() {
                    setStatus('Simulating sensor data collection...', 'success');
                    
                    // Simulate multiple sensor data points
                    const testLocations = [
                        [29.5630, 106.5516, 0], [29.5631, 106.5517, 1], [29.5632, 106.5518, 2],
                        [29.5640, 106.5530, 0], [29.5641, 106.5531, 1],
                        [29.5650, 106.5540, 0], [29.5651, 106.5541, 1], [29.5652, 106.5542, 2]
                    ];
                    
                    let submitted = 0;
                    for (const [lat, lon, elevation] of testLocations) {
                        for (let i = 0; i < 10; i++) {
                            try {
                                const sensorData = generateSimulatedSensorData(lat, lon, elevation);
                                await fetch('/api/submit_sensor_data', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(sensorData)
                                });
                                submitted++;
                            } catch (error) {
                                console.error('Error submitting simulated data:', error);
                            }
                        }
                    }
                    
                    setStatus(`Simulated ${submitted} sensor readings. Rebuilding graph...`, 'success');
                    await rebuildGraph();
                }
                
                function generateSimulatedSensorData(lat, lon, elevation) {
                    const basePressure = 1013.25 - (elevation * 3 * 0.12);
                    return {
                        user_id: `sim_${Date.now()}_${Math.random()}`,
                        gps_lat: lat + (Math.random() - 0.5) * 0.0002,
                        gps_lon: lon + (Math.random() - 0.5) * 0.0002,
                        gps_accuracy: 3 + Math.random() * 12,
                        wifi_signals: Array.from({length: Math.floor(Math.random() * 5) + 1}, (_, i) => ({[`ap_${i}`]: -40 - Math.random() * 40})),
                        bluetooth_beacons: Array.from({length: Math.floor(Math.random() * 3)}, (_, i) => ({[`beacon_${i}`]: -60 - Math.random() * 30})),
                        barometer_pressure: basePressure + (Math.random() - 0.5) * 1,
                        accelerometer_x: (Math.random() - 0.5) * 1,
                        accelerometer_y: (Math.random() - 0.5) * 1,
                        accelerometer_z: -9.81 + (Math.random() - 0.5) * 1,
                        gyroscope_x: (Math.random() - 0.5) * 0.2,
                        gyroscope_y: (Math.random() - 0.5) * 0.2,
                        gyroscope_z: (Math.random() - 0.5) * 0.2,
                        magnetometer_x: 20 + (Math.random() - 0.5) * 10,
                        magnetometer_y: (Math.random() - 0.5) * 10,
                        magnetometer_z: -45 + (Math.random() - 0.5) * 10
                    };
                }
                
                async function loadMapData() {
                    try {
                        const response = await fetch('/api/map_data');
                        const data = await response.json();
                        
                        if (nodesLayer) {
                            map.removeLayer(nodesLayer);
                        }
                        
                        nodesLayer = L.layerGroup().addTo(map);
                        
                        // Add nodes to map
                        data.nodes.forEach(node => {
                            const color = {
                                'entrance': '#00FF00',
                                'junction': '#FF0000',
                                'location': '#0000FF',
                                'custom': '#FF00FF'
                            }[node.type] || '#888888';
                            
                            const marker = L.circleMarker([node.lat, node.lon], {
                                radius: 5,
                                fillColor: color,
                                color: '#000',
                                weight: 1,
                                opacity: 1,
                                fillOpacity: 0.8
                            }).addTo(nodesLayer);
                            
                            marker.bindPopup(`${node.type.toUpperCase()}<br>Elevation: ${node.elevation}<br>ID: ${node.id}`);
                        });
                        
                    } catch (error) {
                        console.error('Error loading map data:', error);
                    }
                }
                
                // Initialize map data on load
                loadMapData();
                
                // Add markers for initial positions
                startMarker = L.marker([29.5630, 106.5516]).addTo(map).bindPopup('Start Location');
                endMarker = L.marker([29.5650, 106.5540]).addTo(map).bindPopup('End Location');
            </script>
        </body>
        </html>
        """
    
    def create_interactive_map(self, routes: List[Route] = None) -> str:
        """Create an interactive Folium map with navigation data."""
        # Center on Chongqing
        m = folium.Map(location=[29.5630, 106.5516], zoom_start=13)
        
        # Add nodes if graph is available
        if self.graph_builder and self.graph_builder.nodes_3d:
            node_cluster = MarkerCluster(name="Navigation Nodes")
            
            for node in self.graph_builder.nodes_3d.values():
                color = {
                    'entrance': 'green',
                    'junction': 'red',
                    'location': 'blue',
                    'custom': 'purple'
                }.get(node.node_type, 'gray')
                
                folium.CircleMarker(
                    location=[node.lat, node.lon],
                    radius=6,
                    popup=f"{node.node_type.title()}<br>Elevation: {node.elevation}<br>ID: {node.id}",
                    color='black',
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(node_cluster)
            
            node_cluster.add_to(m)
        
        # Add routes if provided
        if routes:
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            for i, route in enumerate(routes):
                if route:
                    color = colors[i % len(colors)]
                    
                    # Create route coordinates
                    route_coords = []
                    for step in route.steps:
                        route_coords.append([step.coordinates[0], step.coordinates[1]])
                    
                    if route_coords:
                        folium.PolyLine(
                            locations=route_coords,
                            color=color,
                            weight=4,
                            opacity=0.8,
                            popup=f"Route {i+1}: {route.total_distance:.0f}m, {route.total_duration/60:.1f} min"
                        ).add_to(m)
                        
                        # Add elevation change markers
                        for step in route.steps:
                            if abs(step.elevation_change) > 0:
                                folium.CircleMarker(
                                    location=[step.coordinates[0], step.coordinates[1]],
                                    radius=8,
                                    popup=f"{step.instruction}<br>Elevation change: {step.elevation_change:+d}",
                                    color='black',
                                    fillColor='yellow' if step.elevation_change > 0 else 'cyan',
                                    fillOpacity=0.8
                                ).add_to(m)
        
        return m._repr_html_()
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the navigation application."""
        print(f"Starting 3D Navigation System...")
        print(f"Web interface: http://{host}:{port}")
        print(f"API documentation: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    import sys
    import os
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    app = NavigationApp()
    app.run()