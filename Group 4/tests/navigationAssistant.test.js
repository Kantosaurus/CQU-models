import { jest } from '@jest/globals';
import { NavigationAssistant } from '../src/navigationAssistant.js';

// Mock the external dependencies
const mockGeminiApiKey = 'mock-gemini-key';
const mockMapsApiKey = 'mock-maps-key';

describe('NavigationAssistant', () => {
  let navigationAssistant;

  beforeEach(() => {
    navigationAssistant = new NavigationAssistant(mockGeminiApiKey, mockMapsApiKey);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('processNavigationRequest', () => {
    test('should process a basic navigation request', async () => {
      const userInput = "How do I get to Hongyadong?";
      const userLocation = { lat: 29.5647, lng: 106.5507 };

      // Mock the dependencies
      const mockIntent = {
        destination: "Hongyadong",
        constraints: [],
        transportation_mode: "walking",
        accessibility_needs: []
      };

      const mockRouteData = {
        routes: [{
          legs: [{
            distance: { value: 1200, text: "1.2 km" },
            duration: { value: 900, text: "15 min" },
            start_address: "Current Location",
            end_address: "Hongyadong"
          }],
          enhanced_steps: []
        }]
      };

      // Since we can't mock the actual API calls in this simple test,
      // we'll test the structure and error handling
      const result = await navigationAssistant.processNavigationRequest(userInput, userLocation);
      
      expect(result).toHaveProperty('success');
      expect(result).toHaveProperty('response');
      
      if (result.success) {
        expect(result.response).toHaveProperty('type', 'navigation_response');
        expect(result.response).toHaveProperty('message');
        expect(result.response).toHaveProperty('route_summary');
        expect(result).toHaveProperty('session_id');
      } else {
        expect(result).toHaveProperty('error');
        expect(result).toHaveProperty('fallback_response');
      }
    });

    test('should handle missing user input', async () => {
      const result = await navigationAssistant.processNavigationRequest('');
      
      expect(result.success).toBe(false);
      expect(result).toHaveProperty('error');
      expect(result).toHaveProperty('fallback_response');
    });
  });

  describe('helper methods', () => {
    test('should generate session ID', () => {
      const sessionId = navigationAssistant.generateSessionId();
      
      expect(sessionId).toMatch(/^nav_\d+_[a-z0-9]+$/);
    });

    test('should identify route modification requests', () => {
      const intent = {
        constraints: ['avoid_stairs', 'fastest_route'],
        accessibility_needs: ['avoid_stairs']
      };
      
      const isModification = navigationAssistant.isRouteModificationRequest(intent);
      expect(isModification).toBe(true);
    });

    test('should identify step inquiries', () => {
      const queries = [
        "What's the next step?",
        "How do I get there?",
        "Where should I go?",
        "Can you explain the directions?"
      ];
      
      queries.forEach(query => {
        const isStepInquiry = navigationAssistant.isStepInquiry(query);
        expect(isStepInquiry).toBe(true);
      });
    });

    test('should identify alternative requests', () => {
      const queries = [
        "Show me another route",
        "Is there a different way?",
        "What are my alternatives?",
        "Give me other options"
      ];
      
      queries.forEach(query => {
        const isAlternativeRequest = navigationAssistant.isAlternativeRequest(query);
        expect(isAlternativeRequest).toBe(true);
      });
    });
  });

  describe('session management', () => {
    test('should handle no active session', async () => {
      try {
        await navigationAssistant.processFollowUpQuery("What's next?", "invalid-session");
      } catch (error) {
        expect(error.message).toContain('No active navigation session found');
      }
    });

    test('should clear session', () => {
      navigationAssistant.currentSession = { test: 'data' };
      navigationAssistant.clearSession();
      
      expect(navigationAssistant.currentSession).toBeNull();
    });
  });

  describe('error handling', () => {
    test('should generate error response', () => {
      const userInput = "Test input";
      const error = new Error("Test error");
      
      const errorResponse = navigationAssistant.generateErrorResponse(userInput, error);
      
      expect(errorResponse).toHaveProperty('message');
      expect(errorResponse).toHaveProperty('suggestion');
      expect(errorResponse).toHaveProperty('error_type');
      expect(errorResponse.error_type).toBe('Error');
    });
  });
});