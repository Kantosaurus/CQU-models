import { GoogleGenerativeAI } from '@google/generative-ai';

export class IntentInterpreter {
  constructor(apiKey) {
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = this.genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  }

  async interpretUserIntent(userInput) {
    const prompt = `
You are a navigation assistant AI. Analyze the user's input and extract navigation intent in JSON format.

Extract:
1. destination: the place they want to go
2. constraints: any special requirements (avoid stairs, fastest route, etc.)
3. transportation_mode: preferred method (walking, driving, transit)
4. urgency: how urgent the request is (low, medium, high)

User input: "${userInput}"

Respond ONLY with valid JSON in this format:
{
  "destination": "place name",
  "constraints": ["constraint1", "constraint2"],
  "transportation_mode": "walking|driving|transit",
  "urgency": "low|medium|high",
  "accessibility_needs": ["avoid_stairs", "wheelchair_accessible", "elevator_preferred"]
}
    `;

    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();
      
      // Parse JSON response
      const cleanedText = text.replace(/```json\n?|\n?```/g, '').trim();
      return JSON.parse(cleanedText);
    } catch (error) {
      console.error('Error interpreting user intent:', error);
      // Fallback parsing
      return this.fallbackIntentExtraction(userInput);
    }
  }

  fallbackIntentExtraction(userInput) {
    const input = userInput.toLowerCase();
    
    // Simple pattern matching for fallback
    const destination = this.extractDestination(input);
    const constraints = this.extractConstraints(input);
    const transportationMode = this.extractTransportationMode(input);
    const accessibilityNeeds = this.extractAccessibilityNeeds(input);
    
    return {
      destination: destination || "unknown",
      constraints,
      transportation_mode: transportationMode,
      urgency: input.includes('urgent') || input.includes('quickly') ? 'high' : 'medium',
      accessibility_needs: accessibilityNeeds
    };
  }

  extractDestination(input) {
    // Look for "to" patterns
    const toPattern = /(?:to|get to|go to|find|reach)\s+([^,.\n]+)/i;
    const match = input.match(toPattern);
    return match ? match[1].trim() : null;
  }

  extractConstraints(input) {
    const constraints = [];
    
    if (input.includes('fast') || input.includes('quick')) {
      constraints.push('fastest_route');
    }
    if (input.includes('avoid traffic')) {
      constraints.push('avoid_traffic');
    }
    if (input.includes('scenic') || input.includes('beautiful')) {
      constraints.push('scenic_route');
    }
    
    return constraints;
  }

  extractTransportationMode(input) {
    if (input.includes('drive') || input.includes('car')) return 'driving';
    if (input.includes('walk') || input.includes('on foot')) return 'walking';
    if (input.includes('bus') || input.includes('train') || input.includes('transit')) return 'transit';
    
    return 'walking'; // default
  }

  extractAccessibilityNeeds(input) {
    const needs = [];
    
    if (input.includes('stairs') && (input.includes('avoid') || input.includes('no') || input.includes('without'))) {
      needs.push('avoid_stairs');
    }
    if (input.includes('elevator')) {
      needs.push('elevator_preferred');
    }
    if (input.includes('wheelchair') || input.includes('accessible')) {
      needs.push('wheelchair_accessible');
    }
    
    return needs;
  }

  async generateContextualResponse(routeData, originalIntent) {
    const prompt = `
You are a friendly navigation assistant. Generate a natural, conversational response based on:

Original user request: "${originalIntent.destination}" with needs: ${originalIntent.accessibility_needs.join(', ')}
Route information: ${JSON.stringify(routeData)}

Provide a helpful, context-aware response that:
1. Acknowledges their specific needs (like avoiding stairs)
2. Highlights important navigation points
3. Mentions accessibility features when relevant
4. Uses natural, conversational language

Keep it concise but informative.
    `;

    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error generating contextual response:', error);
      return "I found a route for you. Check the details for the best path to your destination.";
    }
  }
}