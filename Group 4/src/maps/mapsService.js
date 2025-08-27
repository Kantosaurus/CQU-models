import { Client } from '@googlemaps/google-maps-services-js';

export class MapsService {
  constructor(apiKey) {
    this.client = new Client();
    this.apiKey = apiKey;
  }

  async getRoutes(origin, destination, intent) {
    try {
      const travelMode = this.mapTravelMode(intent.transportation_mode);
      
      // Get basic route information
      const directionsResponse = await this.client.directions({
        params: {
          origin: origin,
          destination: destination,
          mode: travelMode,
          alternatives: true,
          key: this.apiKey,
          units: 'metric',
          avoid: this.buildAvoidanceParams(intent.constraints)
        }
      });

      // Get elevation data for accessibility analysis
      const routes = directionsResponse.data.routes;
      const enhancedRoutes = await Promise.all(
        routes.map(route => this.enhanceRouteWithAccessibility(route, intent))
      );

      return {
        status: 'OK',
        routes: enhancedRoutes,
        origin_address: directionsResponse.data.routes[0]?.legs[0]?.start_address,
        destination_address: directionsResponse.data.routes[0]?.legs[0]?.end_address
      };
    } catch (error) {
      console.error('Maps API error:', error);
      throw new Error(`Failed to get routes: ${error.message}`);
    }
  }

  async enhanceRouteWithAccessibility(route, intent) {
    const steps = route.legs[0].steps;
    
    // Analyze each step for accessibility
    const enhancedSteps = await Promise.all(
      steps.map(step => this.analyzeStepAccessibility(step, intent))
    );

    // Calculate accessibility score
    const accessibilityScore = this.calculateAccessibilityScore(enhancedSteps, intent);

    return {
      ...route,
      accessibility_score: accessibilityScore,
      enhanced_steps: enhancedSteps,
      accessibility_summary: this.generateAccessibilitySummary(enhancedSteps, intent)
    };
  }

  async analyzeStepAccessibility(step, intent) {
    const analysis = {
      ...step,
      accessibility_features: [],
      warnings: [],
      alternatives: []
    };

    // Analyze instruction text for stairs/elevation
    const instruction = step.html_instructions.toLowerCase();
    
    if (instruction.includes('stairs') || instruction.includes('steps')) {
      analysis.accessibility_features.push('stairs_present');
      
      if (intent.accessibility_needs.includes('avoid_stairs')) {
        analysis.warnings.push('This step contains stairs');
        analysis.alternatives.push('Look for elevator or ramp alternatives');
      }
    }

    if (instruction.includes('elevator')) {
      analysis.accessibility_features.push('elevator_available');
    }

    if (instruction.includes('ramp')) {
      analysis.accessibility_features.push('ramp_available');
    }

    // Get elevation data if available
    try {
      const elevationData = await this.getElevationProfile(step);
      analysis.elevation_change = elevationData.elevation_change;
      analysis.grade = elevationData.grade;
      
      if (Math.abs(elevationData.grade) > 5) {
        analysis.warnings.push(`Steep ${elevationData.grade > 0 ? 'uphill' : 'downhill'} section`);
      }
    } catch (error) {
      // Elevation data not critical
      console.warn('Could not get elevation data:', error.message);
    }

    return analysis;
  }

  async getElevationProfile(step) {
    try {
      const startLocation = `${step.start_location.lat},${step.start_location.lng}`;
      const endLocation = `${step.end_location.lat},${step.end_location.lng}`;

      const elevationResponse = await this.client.elevation({
        params: {
          locations: [startLocation, endLocation],
          key: this.apiKey
        }
      });

      const results = elevationResponse.data.results;
      if (results.length >= 2) {
        const elevationChange = results[1].elevation - results[0].elevation;
        const distance = step.distance.value; // in meters
        const grade = distance > 0 ? (elevationChange / distance) * 100 : 0;

        return {
          elevation_change: elevationChange,
          grade: grade,
          start_elevation: results[0].elevation,
          end_elevation: results[1].elevation
        };
      }

      return { elevation_change: 0, grade: 0 };
    } catch (error) {
      throw new Error(`Elevation API error: ${error.message}`);
    }
  }

  calculateAccessibilityScore(enhancedSteps, intent) {
    let score = 100; // Start with perfect score
    
    for (const step of enhancedSteps) {
      // Deduct points for accessibility issues
      if (step.accessibility_features.includes('stairs_present') && 
          intent.accessibility_needs.includes('avoid_stairs')) {
        score -= 20;
      }
      
      if (step.grade && Math.abs(step.grade) > 5) {
        score -= Math.min(Math.abs(step.grade), 30); // Cap deduction at 30 points
      }
      
      if (step.warnings.length > 0) {
        score -= step.warnings.length * 5;
      }
      
      // Add points for positive features
      if (step.accessibility_features.includes('elevator_available') ||
          step.accessibility_features.includes('ramp_available')) {
        score += 10;
      }
    }
    
    return Math.max(0, Math.min(100, score)); // Clamp between 0-100
  }

  generateAccessibilitySummary(enhancedSteps, intent) {
    const summary = {
      total_stairs: 0,
      elevators_available: 0,
      ramps_available: 0,
      steep_sections: 0,
      warnings: [],
      recommendations: []
    };

    for (const step of enhancedSteps) {
      if (step.accessibility_features.includes('stairs_present')) {
        summary.total_stairs++;
      }
      if (step.accessibility_features.includes('elevator_available')) {
        summary.elevators_available++;
      }
      if (step.accessibility_features.includes('ramp_available')) {
        summary.ramps_available++;
      }
      if (step.grade && Math.abs(step.grade) > 5) {
        summary.steep_sections++;
      }
      
      summary.warnings.push(...step.warnings);
    }

    // Generate recommendations
    if (intent.accessibility_needs.includes('avoid_stairs') && summary.total_stairs > 0) {
      summary.recommendations.push(`This route has ${summary.total_stairs} sections with stairs. Consider alternative routes.`);
    }
    
    if (summary.elevators_available > 0) {
      summary.recommendations.push(`${summary.elevators_available} elevator(s) available on this route.`);
    }

    return summary;
  }

  mapTravelMode(mode) {
    const modeMap = {
      'walking': 'walking',
      'driving': 'driving', 
      'transit': 'transit'
    };
    
    return modeMap[mode] || 'walking';
  }

  buildAvoidanceParams(constraints) {
    const avoidParams = [];
    
    if (constraints.includes('avoid_traffic')) {
      avoidParams.push('traffic');
    }
    if (constraints.includes('avoid_tolls')) {
      avoidParams.push('tolls');
    }
    if (constraints.includes('avoid_highways')) {
      avoidParams.push('highways');
    }
    
    return avoidParams;
  }

  async searchPlaces(query, location = null) {
    try {
      const params = {
        query: query,
        key: this.apiKey
      };
      
      if (location) {
        params.location = `${location.lat},${location.lng}`;
        params.radius = 5000; // 5km radius
      }

      const response = await this.client.textSearch({ params });
      return response.data.results;
    } catch (error) {
      console.error('Places search error:', error);
      throw new Error(`Failed to search places: ${error.message}`);
    }
  }
}