import { IntentInterpreter } from './ai/intentInterpreter.js';
import { ResponseGenerator } from './ai/responseGenerator.js';
import { MapsService } from './maps/mapsService.js';
import { RouteProcessor } from './utils/routeProcessor.js';

export class NavigationAssistant {
  constructor(geminiApiKey, mapsApiKey) {
    this.intentInterpreter = new IntentInterpreter(geminiApiKey);
    this.responseGenerator = new ResponseGenerator(geminiApiKey);
    this.mapsService = new MapsService(mapsApiKey);
    this.routeProcessor = new RouteProcessor();
    
    this.currentSession = null;
  }

  async processNavigationRequest(userInput, userLocation = null) {
    try {
      console.log('🤖 Processing navigation request:', userInput);
      
      // Step 1: User intent → AI
      console.log('📝 Interpreting user intent...');
      const userIntent = await this.intentInterpreter.interpretUserIntent(userInput);
      console.log('✅ Intent interpreted:', userIntent);

      // Step 2: AI → Maps API
      console.log('🗺️  Calling Maps API...');
      const origin = userLocation || 'Current Location';
      const routeData = await this.mapsService.getRoutes(origin, userIntent.destination, userIntent);
      console.log('✅ Routes retrieved:', routeData.routes.length, 'routes found');

      // Step 3: Maps API → AI (Process and analyze)
      console.log('🔄 Processing route data...');
      const processedRouteData = this.routeProcessor.processRoutes(routeData, userIntent);
      console.log('✅ Routes processed with accessibility scores');

      // Step 4: AI interprets & rephrases
      console.log('💬 Generating natural language response...');
      const aiResponse = await this.responseGenerator.generateNavigationResponse(
        processedRouteData,
        userIntent,
        userLocation
      );
      console.log('✅ Response generated');

      // Step 5: Create session for potential follow-up
      this.currentSession = {
        userIntent,
        processedRouteData,
        currentStepIndex: 0,
        startTime: new Date(),
        userLocation
      };

      return {
        success: true,
        response: aiResponse,
        session_id: this.generateSessionId(),
        debug_info: {
          intent: userIntent,
          routes_found: processedRouteData.routes.length,
          best_accessibility_score: processedRouteData.routes[0]?.accessibility_score
        }
      };

    } catch (error) {
      console.error('❌ Navigation processing error:', error);
      
      return {
        success: false,
        error: error.message,
        fallback_response: this.generateErrorResponse(userInput, error)
      };
    }
  }

  async processFollowUpQuery(query, sessionId) {
    if (!this.currentSession) {
      throw new Error('No active navigation session found');
    }

    const intent = await this.intentInterpreter.interpretUserIntent(query);
    
    // Handle different types of follow-up queries
    if (this.isRouteModificationRequest(intent)) {
      return await this.modifyCurrentRoute(intent);
    }
    
    if (this.isStepInquiry(query)) {
      return this.provideStepDetails(query);
    }
    
    if (this.isAlternativeRequest(query)) {
      return this.provideAlternativeRoutes();
    }
    
    // General follow-up response
    return await this.responseGenerator.generateContextualResponse(
      this.currentSession.processedRouteData.routes[0],
      this.currentSession.userIntent,
      this.currentSession.processedRouteData.recommendation
    );
  }

  async startLiveNavigation(sessionId) {
    if (!this.currentSession) {
      throw new Error('No active navigation session found');
    }

    const bestRoute = this.currentSession.processedRouteData.routes[0];
    const firstStep = bestRoute.processed_steps[0];
    const secondStep = bestRoute.processed_steps[1];

    const liveGuidance = await this.responseGenerator.generateLiveGuidance(
      firstStep,
      secondStep,
      this.currentSession.userLocation,
      this.currentSession.userIntent
    );

    return {
      type: 'live_navigation_start',
      current_step: firstStep,
      live_instruction: liveGuidance,
      progress: {
        current_step: 1,
        total_steps: bestRoute.processed_steps.length,
        percentage: Math.round((1 / bestRoute.processed_steps.length) * 100)
      }
    };
  }

  async updateNavigationProgress(currentLocation, sessionId) {
    if (!this.currentSession) {
      throw new Error('No active navigation session found');
    }

    // This would typically involve comparing current location with route steps
    // For now, we'll simulate step progression
    const bestRoute = this.currentSession.processedRouteData.routes[0];
    const currentStepIndex = this.currentSession.currentStepIndex;
    
    if (currentStepIndex < bestRoute.processed_steps.length - 1) {
      this.currentSession.currentStepIndex += 1;
      const newCurrentStep = bestRoute.processed_steps[this.currentSession.currentStepIndex];
      const nextStep = bestRoute.processed_steps[this.currentSession.currentStepIndex + 1];

      const liveGuidance = await this.responseGenerator.generateLiveGuidance(
        newCurrentStep,
        nextStep,
        currentLocation,
        this.currentSession.userIntent
      );

      return {
        type: 'navigation_update',
        current_step: newCurrentStep,
        live_instruction: liveGuidance,
        progress: {
          current_step: this.currentSession.currentStepIndex + 1,
          total_steps: bestRoute.processed_steps.length,
          percentage: Math.round(((this.currentSession.currentStepIndex + 1) / bestRoute.processed_steps.length) * 100)
        }
      };
    } else {
      return {
        type: 'navigation_complete',
        message: `🎉 You've arrived at ${this.currentSession.userIntent.destination}! I hope the route worked well for your accessibility needs.`,
        journey_summary: this.generateJourneySummary()
      };
    }
  }

  async searchNearbyPlaces(query, userLocation = null, filters = {}) {
    try {
      const places = await this.mapsService.searchPlaces(query, userLocation);
      
      // Filter places based on accessibility if specified
      if (filters.accessibility_required) {
        // This would need additional API calls to check accessibility features
        // For now, we'll return all places with a note about accessibility
      }

      return {
        success: true,
        places: places.slice(0, 10), // Limit to top 10 results
        accessibility_note: filters.accessibility_required 
          ? "Please verify accessibility features directly with venues"
          : null
      };
    } catch (error) {
      console.error('Places search error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // Helper methods

  isRouteModificationRequest(intent) {
    return intent.constraints.some(constraint => 
      ['avoid_stairs', 'fastest_route', 'avoid_traffic'].includes(constraint)
    ) || intent.accessibility_needs.length > 0;
  }

  isStepInquiry(query) {
    const stepKeywords = ['step', 'instruction', 'direction', 'how', 'where'];
    return stepKeywords.some(keyword => query.toLowerCase().includes(keyword));
  }

  isAlternativeRequest(query) {
    const altKeywords = ['alternative', 'other route', 'different way', 'another option'];
    return altKeywords.some(keyword => query.toLowerCase().includes(keyword));
  }

  async modifyCurrentRoute(newIntent) {
    // Re-process with new constraints
    const origin = this.currentSession.userLocation || 'Current Location';
    const routeData = await this.mapsService.getRoutes(
      origin, 
      newIntent.destination || this.currentSession.userIntent.destination, 
      newIntent
    );

    const processedRouteData = this.routeProcessor.processRoutes(routeData, newIntent);
    
    // Update current session
    this.currentSession.userIntent = { ...this.currentSession.userIntent, ...newIntent };
    this.currentSession.processedRouteData = processedRouteData;
    this.currentSession.currentStepIndex = 0;

    const aiResponse = await this.responseGenerator.generateNavigationResponse(
      processedRouteData,
      this.currentSession.userIntent,
      this.currentSession.userLocation
    );

    return {
      success: true,
      response: aiResponse,
      modification_note: "Route updated based on your new preferences"
    };
  }

  provideStepDetails(query) {
    const bestRoute = this.currentSession.processedRouteData.routes[0];
    const currentStep = bestRoute.processed_steps[this.currentSession.currentStepIndex];
    
    return {
      type: 'step_details',
      step: currentStep,
      accessibility_info: {
        warnings: currentStep.accessibility_warnings,
        suggestions: currentStep.accessibility_suggestions,
        difficulty_level: currentStep.difficulty_level
      }
    };
  }

  provideAlternativeRoutes() {
    const alternatives = this.currentSession.processedRouteData.routes.slice(1, 4);
    
    return {
      type: 'alternative_routes',
      routes: alternatives.map(route => ({
        summary: route.user_friendly_summary,
        accessibility_score: route.accessibility_score,
        trade_offs: this.routeProcessor.generateTradeOffs ? 
          this.routeProcessor.generateTradeOffs(route) : []
      })),
      recommendation: "Consider these alternatives based on your preferences"
    };
  }

  generateJourneySummary() {
    const session = this.currentSession;
    const duration = new Date() - session.startTime;
    
    return {
      destination: session.userIntent.destination,
      journey_time: Math.round(duration / 60000), // minutes
      accessibility_score: session.processedRouteData.routes[0].accessibility_score,
      steps_completed: session.currentStepIndex + 1,
      accessibility_features_used: session.processedRouteData.routes[0].accessibility_summary
    };
  }

  generateErrorResponse(userInput, error) {
    return {
      message: "I'm sorry, I had trouble processing your navigation request. Let me try to help you differently.",
      suggestion: "Could you rephrase your destination or let me know if you have specific accessibility needs?",
      error_type: error.name || 'UnknownError'
    };
  }

  generateSessionId() {
    return 'nav_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  clearSession() {
    this.currentSession = null;
  }
}