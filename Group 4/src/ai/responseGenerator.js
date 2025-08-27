import { GoogleGenerativeAI } from '@google/generative-ai';

export class ResponseGenerator {
  constructor(apiKey) {
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = this.genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  }

  async generateNavigationResponse(processedRouteData, originalIntent, userLocation = null) {
    const bestRoute = processedRouteData.routes[0];
    
    const contextualResponse = await this.generateContextualResponse(
      bestRoute, 
      originalIntent, 
      processedRouteData.recommendation
    );

    return {
      type: 'navigation_response',
      message: contextualResponse,
      route_summary: this.generateQuickSummary(bestRoute, originalIntent),
      step_by_step: this.generateStepByStepInstructions(bestRoute, originalIntent),
      accessibility_info: this.formatAccessibilityInfo(bestRoute, originalIntent),
      alternatives: this.generateAlternatives(processedRouteData.routes.slice(1, 3)),
      quick_tips: this.generateQuickTips(bestRoute, originalIntent)
    };
  }

  async generateContextualResponse(route, originalIntent, recommendation) {
    const prompt = this.buildContextualPrompt(route, originalIntent, recommendation);
    
    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error generating contextual response:', error);
      return this.generateFallbackResponse(route, originalIntent);
    }
  }

  buildContextualPrompt(route, originalIntent, recommendation) {
    return `
You are a helpful, friendly navigation assistant. Generate a natural, conversational response for a user's navigation request.

USER REQUEST: "${originalIntent.destination}" 
USER NEEDS: ${originalIntent.accessibility_needs.join(', ') || 'none specified'}
TRANSPORTATION: ${originalIntent.transportation_mode}

ROUTE INFORMATION:
- Distance: ${this.formatDistance(route.legs[0].distance.value)}
- Duration: ${route.time_estimate_adjusted?.text || route.legs[0].duration.text}
- Accessibility Score: ${route.accessibility_score}/100
- ${route.accessibility_summary.total_stairs} stair section(s)
- ${route.accessibility_summary.elevators_available} elevator(s) available

RECOMMENDATION: ${recommendation.reason}

Generate a response that:
1. Acknowledges their specific destination and needs
2. Provides the key route information in an encouraging way
3. Highlights relevant accessibility features
4. Gives one practical tip for the journey
5. Uses a warm, helpful tone (like a local friend helping out)

Keep it conversational and around 3-4 sentences. Don't be overly formal.

EXAMPLE TONE: "Great choice! I found a nice route to Hongyadong that should work well for you..."
    `;
  }

  generateFallbackResponse(route, originalIntent) {
    const destination = originalIntent.destination;
    const duration = route.time_estimate_adjusted?.text || route.legs[0].duration.text;
    const distance = this.formatDistance(route.legs[0].distance.value);
    
    let response = `I found a good route to ${destination}! It's ${distance} and should take about ${duration}.`;
    
    if (originalIntent.accessibility_needs.includes('avoid_stairs') && 
        route.accessibility_summary.total_stairs === 0) {
      response += " Perfect news - this route is completely stair-free!";
    } else if (route.accessibility_summary.elevators_available > 0) {
      response += ` There ${route.accessibility_summary.elevators_available === 1 ? 'is' : 'are'} ${route.accessibility_summary.elevators_available} elevator${route.accessibility_summary.elevators_available > 1 ? 's' : ''} along the way to help with accessibility.`;
    }
    
    return response;
  }

  generateQuickSummary(route, originalIntent) {
    const summary = {
      destination: originalIntent.destination,
      distance: this.formatDistance(route.legs[0].distance.value),
      duration: route.time_estimate_adjusted?.text || route.legs[0].duration.text,
      accessibility_score: route.accessibility_score,
      transportation_mode: originalIntent.transportation_mode,
      key_features: []
    };

    if (route.accessibility_summary.total_stairs === 0) {
      summary.key_features.push("No stairs");
    }
    
    if (route.accessibility_summary.elevators_available > 0) {
      summary.key_features.push(`${route.accessibility_summary.elevators_available} elevator(s)`);
    }
    
    if (route.accessibility_score > 85) {
      summary.key_features.push("Excellent accessibility");
    }

    return summary;
  }

  generateStepByStepInstructions(route, originalIntent) {
    return route.processed_steps.map((step, index) => {
      const instruction = {
        step_number: index + 1,
        instruction: step.enhanced_instruction.replace(/<[^>]*>/g, ''), // Remove HTML tags
        distance: step.distance.text,
        duration: step.adjusted_duration?.text || step.duration.text,
        accessibility_level: this.mapDifficultyToText(step.difficulty_level),
        warnings: step.accessibility_warnings,
        tips: step.accessibility_suggestions
      };

      // Add contextual guidance
      if (step.accessibility_features.includes('stairs_present') && 
          originalIntent.accessibility_needs.includes('avoid_stairs')) {
        instruction.special_note = "⚠️ Look for elevator or ramp alternatives here";
      }

      if (step.accessibility_features.includes('elevator_available')) {
        instruction.special_note = "✅ Elevator access available";
      }

      return instruction;
    });
  }

  formatAccessibilityInfo(route, originalIntent) {
    const info = {
      overall_score: route.accessibility_score,
      assessment: route.accessibility_summary,
      user_specific_notes: []
    };

    if (originalIntent.accessibility_needs.includes('avoid_stairs')) {
      if (route.accessibility_summary.total_stairs === 0) {
        info.user_specific_notes.push("✅ Perfect! This route has no stairs at all.");
      } else {
        info.user_specific_notes.push(
          `⚠️ This route has ${route.accessibility_summary.total_stairs} stair section(s). Look for elevator alternatives.`
        );
      }
    }

    if (originalIntent.accessibility_needs.includes('elevator_preferred')) {
      if (route.accessibility_summary.elevators_available > 0) {
        info.user_specific_notes.push(
          `✅ Great! ${route.accessibility_summary.elevators_available} elevator(s) available on this route.`
        );
      } else {
        info.user_specific_notes.push("⚠️ No elevators detected on this route.");
      }
    }

    return info;
  }

  generateAlternatives(alternativeRoutes) {
    return alternativeRoutes.map((route, index) => ({
      option_number: index + 2, // Starting from 2 since main route is 1
      summary: route.user_friendly_summary,
      accessibility_score: route.accessibility_score,
      trade_offs: this.generateTradeOffs(route),
      best_for: this.determineBestUseCase(route)
    }));
  }

  generateTradeOffs(route) {
    const tradeOffs = [];
    
    if (route.accessibility_score < 70) {
      tradeOffs.push("Lower accessibility score");
    }
    
    if (route.legs[0].duration.value > 3600) { // More than 1 hour
      tradeOffs.push("Longer travel time");
    }
    
    if (route.accessibility_summary.total_stairs > 2) {
      tradeOffs.push("Multiple stair sections");
    }
    
    return tradeOffs;
  }

  determineBestUseCase(route) {
    if (route.accessibility_score > 85) return "Users with high accessibility needs";
    if (route.legs[0].duration.value < 1800) return "Users prioritizing speed";
    if (route.legs[0].distance.value < 1000) return "Short distance travelers";
    return "Alternative option";
  }

  generateQuickTips(route, originalIntent) {
    const tips = [];
    
    // General navigation tips
    if (originalIntent.transportation_mode === 'walking') {
      tips.push("💡 Wear comfortable walking shoes");
    }
    
    // Accessibility-specific tips
    if (originalIntent.accessibility_needs.includes('avoid_stairs')) {
      tips.push("🚶‍♀️ Always check for elevator signs at stations and buildings");
      
      if (route.accessibility_summary.elevators_available > 0) {
        tips.push("🏢 Elevators are available - look for accessibility signage");
      }
    }
    
    // Weather considerations
    if (route.legs[0].distance.value > 500) { // More than 500m walking
      tips.push("☂️ Check weather conditions before starting your journey");
    }
    
    // Time-specific tips
    const currentHour = new Date().getHours();
    if (currentHour >= 7 && currentHour <= 9) {
      tips.push("🚇 Rush hour - public transit and elevators may be crowded");
    }
    
    // Route-specific tips
    if (route.accessibility_summary.steep_sections > 0) {
      tips.push("⛰️ Some uphill sections - take breaks as needed");
    }
    
    return tips.slice(0, 3); // Limit to 3 tips max
  }

  async generateLiveGuidance(currentStep, nextStep, userLocation, originalIntent) {
    const prompt = `
Generate a short, real-time navigation instruction for a user currently walking.

CURRENT STEP: ${currentStep.enhanced_instruction}
NEXT STEP: ${nextStep ? nextStep.enhanced_instruction : 'Destination reached'}
USER NEEDS: ${originalIntent.accessibility_needs.join(', ')}

Create a brief, encouraging instruction (1-2 sentences) that:
1. Tells them what to do now
2. Mentions any accessibility features they should look for
3. Uses present tense and active voice

Example: "Continue straight for 100 meters. Look for the elevator on your right to avoid the stairs ahead."
    `;

    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error generating live guidance:', error);
      return this.generateFallbackLiveGuidance(currentStep, originalIntent);
    }
  }

  generateFallbackLiveGuidance(currentStep, originalIntent) {
    let guidance = currentStep.enhanced_instruction.replace(/<[^>]*>/g, '');
    
    if (currentStep.accessibility_features.includes('elevator_available')) {
      guidance += " Look for elevator access.";
    }
    
    if (currentStep.accessibility_features.includes('stairs_present') && 
        originalIntent.accessibility_needs.includes('avoid_stairs')) {
      guidance += " Check for ramp or elevator alternatives.";
    }
    
    return guidance;
  }

  mapDifficultyToText(level) {
    const difficultyMap = {
      1: "Easy",
      2: "Moderate", 
      3: "Challenging",
      4: "Difficult",
      5: "Very Difficult"
    };
    
    return difficultyMap[level] || "Moderate";
  }

  formatDistance(meters) {
    if (meters < 1000) return `${meters} m`;
    return `${(meters / 1000).toFixed(1)} km`;
  }
}