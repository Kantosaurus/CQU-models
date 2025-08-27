export class RouteProcessor {
  constructor() {
    this.accessibilityKeywords = {
      stairs: ['stairs', 'steps', 'staircase', 'stairway'],
      elevator: ['elevator', 'lift'],
      escalator: ['escalator', 'moving stairs'],
      ramp: ['ramp', 'slope', 'incline'],
      bridge: ['bridge', 'overpass', 'walkway'],
      tunnel: ['tunnel', 'underpass', 'subway']
    };
  }

  processRoutes(routeData, userIntent) {
    const processedRoutes = routeData.routes.map(route => 
      this.processIndividualRoute(route, userIntent)
    );

    // Sort routes by accessibility score and user preferences
    const sortedRoutes = this.sortRoutesByPreference(processedRoutes, userIntent);

    return {
      ...routeData,
      routes: sortedRoutes,
      recommendation: this.generateRouteRecommendation(sortedRoutes, userIntent),
      accessibility_summary: this.generateOverallAccessibilitySummary(sortedRoutes)
    };
  }

  processIndividualRoute(route, userIntent) {
    const processedSteps = route.enhanced_steps.map(step => 
      this.enhanceStepInstructions(step, userIntent)
    );

    return {
      ...route,
      processed_steps: processedSteps,
      user_friendly_summary: this.generateUserFriendlySummary(route, userIntent),
      time_estimate_adjusted: this.adjustTimeForAccessibility(route, userIntent)
    };
  }

  enhanceStepInstructions(step, userIntent) {
    let enhancedInstruction = step.html_instructions;
    const warnings = [...step.warnings];
    const suggestions = [...step.alternatives];

    // Add accessibility-specific guidance
    if (userIntent.accessibility_needs.includes('avoid_stairs')) {
      enhancedInstruction = this.addStairAvoidanceGuidance(enhancedInstruction);
    }

    if (userIntent.accessibility_needs.includes('elevator_preferred')) {
      enhancedInstruction = this.addElevatorGuidance(enhancedInstruction);
    }

    // Add contextual timing adjustments
    const adjustedDuration = this.adjustStepDuration(step, userIntent);

    return {
      ...step,
      enhanced_instruction: enhancedInstruction,
      original_instruction: step.html_instructions,
      accessibility_warnings: warnings,
      accessibility_suggestions: suggestions,
      adjusted_duration: adjustedDuration,
      difficulty_level: this.calculateStepDifficulty(step, userIntent)
    };
  }

  addStairAvoidanceGuidance(instruction) {
    const lowerInstruction = instruction.toLowerCase();
    
    for (const keyword of this.accessibilityKeywords.stairs) {
      if (lowerInstruction.includes(keyword)) {
        return instruction + ' <strong>⚠️ Note: Look for elevator or ramp alternatives to avoid stairs.</strong>';
      }
    }

    return instruction;
  }

  addElevatorGuidance(instruction) {
    const lowerInstruction = instruction.toLowerCase();
    
    for (const keyword of this.accessibilityKeywords.elevator) {
      if (lowerInstruction.includes(keyword)) {
        return instruction + ' <strong>✅ Elevator available - perfect for accessibility needs!</strong>';
      }
    }

    // Look for buildings or stations where elevators might be available
    if (lowerInstruction.includes('station') || 
        lowerInstruction.includes('building') || 
        lowerInstruction.includes('mall')) {
      return instruction + ' <em>💡 Check for elevator access in this area.</em>';
    }

    return instruction;
  }

  adjustStepDuration(step, userIntent) {
    let baseDuration = step.duration.value; // seconds
    let multiplier = 1.0;

    // Adjust for accessibility needs
    if (userIntent.accessibility_needs.includes('avoid_stairs') && 
        step.accessibility_features.includes('stairs_present')) {
      multiplier *= 1.5; // Extra time to find alternatives
    }

    if (step.grade && Math.abs(step.grade) > 3) {
      multiplier *= 1.2; // Extra time for steep sections
    }

    // Adjust for transportation mode
    if (userIntent.transportation_mode === 'walking' && step.distance.value > 500) {
      multiplier *= 1.1; // Conservative estimate for longer walks
    }

    return {
      value: Math.round(baseDuration * multiplier),
      text: this.formatDuration(Math.round(baseDuration * multiplier)),
      adjustment_factor: multiplier,
      original_duration: step.duration
    };
  }

  calculateStepDifficulty(step, userIntent) {
    let difficulty = 1; // Base difficulty (1-5 scale)

    // Increase difficulty for stairs if user wants to avoid them
    if (step.accessibility_features.includes('stairs_present') && 
        userIntent.accessibility_needs.includes('avoid_stairs')) {
      difficulty += 2;
    }

    // Increase difficulty for steep grades
    if (step.grade) {
      if (Math.abs(step.grade) > 10) difficulty += 2;
      else if (Math.abs(step.grade) > 5) difficulty += 1;
    }

    // Decrease difficulty for accessibility features
    if (step.accessibility_features.includes('elevator_available') ||
        step.accessibility_features.includes('ramp_available')) {
      difficulty -= 1;
    }

    return Math.max(1, Math.min(5, difficulty));
  }

  sortRoutesByPreference(routes, userIntent) {
    return routes.sort((a, b) => {
      // Primary sort: accessibility score
      if (userIntent.accessibility_needs.length > 0) {
        const scoreDiff = b.accessibility_score - a.accessibility_score;
        if (Math.abs(scoreDiff) > 10) return scoreDiff;
      }

      // Secondary sort: based on constraints
      if (userIntent.constraints.includes('fastest_route')) {
        return a.legs[0].duration.value - b.legs[0].duration.value;
      }

      // Default: balance accessibility and time
      const aWeight = a.accessibility_score * 0.7 + (1 / a.legs[0].duration.value) * 1000 * 0.3;
      const bWeight = b.accessibility_score * 0.7 + (1 / b.legs[0].duration.value) * 1000 * 0.3;
      
      return bWeight - aWeight;
    });
  }

  generateRouteRecommendation(routes, userIntent) {
    if (!routes.length) return null;

    const bestRoute = routes[0];
    const recommendation = {
      route_index: 0,
      reason: this.generateRecommendationReason(bestRoute, userIntent),
      confidence: this.calculateConfidence(bestRoute, userIntent),
      key_highlights: this.extractKeyHighlights(bestRoute, userIntent)
    };

    return recommendation;
  }

  generateRecommendationReason(route, userIntent) {
    const reasons = [];

    if (route.accessibility_score > 80) {
      reasons.push("excellent accessibility features");
    } else if (route.accessibility_score > 60) {
      reasons.push("good accessibility with minor challenges");
    }

    if (userIntent.accessibility_needs.includes('avoid_stairs') && 
        route.accessibility_summary.total_stairs === 0) {
      reasons.push("completely stair-free");
    }

    if (route.accessibility_summary.elevators_available > 0) {
      reasons.push(`${route.accessibility_summary.elevators_available} elevator(s) available`);
    }

    if (userIntent.constraints.includes('fastest_route')) {
      reasons.push("fastest option available");
    }

    return reasons.length > 0 
      ? `Recommended because it offers ${reasons.join(', ')}.`
      : "Best overall balance of accessibility and efficiency.";
  }

  calculateConfidence(route, userIntent) {
    let confidence = 0.7; // Base confidence

    // Increase confidence for good accessibility match
    if (route.accessibility_score > 80) confidence += 0.2;
    if (route.accessibility_score > 60) confidence += 0.1;

    // Increase confidence if user needs are well met
    if (userIntent.accessibility_needs.includes('avoid_stairs') && 
        route.accessibility_summary.total_stairs === 0) {
      confidence += 0.15;
    }

    return Math.min(1.0, confidence);
  }

  extractKeyHighlights(route, userIntent) {
    const highlights = [];

    if (route.accessibility_score > 90) {
      highlights.push("🌟 Excellent accessibility");
    }

    if (route.accessibility_summary.total_stairs === 0) {
      highlights.push("✅ No stairs");
    }

    if (route.accessibility_summary.elevators_available > 0) {
      highlights.push(`🏢 ${route.accessibility_summary.elevators_available} elevator(s)`);
    }

    if (route.accessibility_summary.steep_sections === 0) {
      highlights.push("📏 Gentle slopes");
    }

    const totalDuration = route.legs.reduce((sum, leg) => sum + leg.duration.value, 0);
    highlights.push(`⏱️ ${this.formatDuration(totalDuration)}`);

    return highlights;
  }

  generateOverallAccessibilitySummary(routes) {
    if (!routes.length) return null;

    const bestRoute = routes[0];
    const summary = {
      best_accessibility_score: bestRoute.accessibility_score,
      total_routes_analyzed: routes.length,
      accessibility_features_available: this.consolidateAccessibilityFeatures(routes),
      overall_recommendation: this.generateOverallRecommendation(bestRoute)
    };

    return summary;
  }

  consolidateAccessibilityFeatures(routes) {
    const features = new Set();
    
    routes.forEach(route => {
      route.enhanced_steps.forEach(step => {
        step.accessibility_features.forEach(feature => features.add(feature));
      });
    });

    return Array.from(features);
  }

  generateOverallRecommendation(bestRoute) {
    const score = bestRoute.accessibility_score;
    
    if (score > 90) return "Excellent accessibility - highly recommended";
    if (score > 75) return "Good accessibility with minor considerations";
    if (score > 60) return "Moderate accessibility - some challenges present";
    if (score > 40) return "Limited accessibility - consider alternatives";
    return "Significant accessibility challenges - alternative routes recommended";
  }

  generateUserFriendlySummary(route, userIntent) {
    const totalDistance = route.legs.reduce((sum, leg) => sum + leg.distance.value, 0);
    const totalDuration = route.legs.reduce((sum, leg) => sum + leg.duration.value, 0);
    
    let summary = `${this.formatDistance(totalDistance)} • ${this.formatDuration(totalDuration)}`;
    
    if (route.accessibility_score < 70) {
      summary += " • ⚠️ Some accessibility challenges";
    } else if (route.accessibility_score > 85) {
      summary += " • ✅ Great accessibility";
    }

    return summary;
  }

  adjustTimeForAccessibility(route, userIntent) {
    const originalDuration = route.legs.reduce((sum, leg) => sum + leg.duration.value, 0);
    let adjustmentFactor = 1.0;

    // Add time for accessibility considerations
    if (userIntent.accessibility_needs.length > 0) {
      adjustmentFactor += 0.1; // Base 10% increase for accessibility planning
    }

    if (route.accessibility_summary.total_stairs > 0 && 
        userIntent.accessibility_needs.includes('avoid_stairs')) {
      adjustmentFactor += 0.2; // 20% more time to find alternatives
    }

    const adjustedDuration = Math.round(originalDuration * adjustmentFactor);

    return {
      value: adjustedDuration,
      text: this.formatDuration(adjustedDuration),
      original: route.legs[0].duration,
      adjustment_factor: adjustmentFactor
    };
  }

  formatDuration(seconds) {
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min`;
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  }

  formatDistance(meters) {
    if (meters < 1000) return `${meters} m`;
    return `${(meters / 1000).toFixed(1)} km`;
  }
}