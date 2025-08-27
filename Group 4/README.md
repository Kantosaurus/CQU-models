# 🤖 AI Navigation Assistant - Group 4

An intelligent navigation system that combines **Google Gemini AI** with **Google Maps API** to provide accessibility-aware, natural language navigation assistance.

## 🌟 Key Features

### 🔄 The AI-Maps Processing Loop
1. **User Intent → AI**: Natural language input is interpreted by Gemini AI
2. **AI → Maps API**: Structured queries are sent to Google Maps
3. **Maps API → AI**: Route data is processed and enhanced with accessibility analysis
4. **AI Interprets & Rephrases**: Raw data becomes human-friendly, context-aware guidance
5. **User Output**: Clean, personalized navigation with accessibility considerations

### ✨ Core Capabilities
- 🧠 **Smart Intent Recognition**: Understands requests like "get to Hongyadong without stairs"
- ♿ **Accessibility Analysis**: Evaluates routes for stairs, elevators, steep grades
- 🗣️ **Natural Language Responses**: Converts technical route data into friendly guidance
- 🚶‍♀️ **Live Navigation**: Real-time step-by-step instructions
- 🔄 **Follow-up Queries**: Handles questions about routes and alternatives
- 📍 **Place Search**: Find nearby locations with accessibility considerations

## 🏗️ Architecture

```
group4/
├── src/
│   ├── ai/
│   │   ├── intentInterpreter.js    # Gemini AI intent processing
│   │   └── responseGenerator.js    # Natural language generation
│   ├── maps/
│   │   └── mapsService.js          # Google Maps API integration
│   ├── utils/
│   │   └── routeProcessor.js       # Route analysis & accessibility scoring
│   ├── navigationAssistant.js     # Main orchestration class
│   └── index.js                    # Express.js server
├── config/
├── tests/
└── package.json
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Google Gemini AI API key
- Google Maps API key with the following APIs enabled:
  - Directions API
  - Places API  
  - Elevation API
  - Geocoding API

### Installation

1. **Clone and install dependencies:**
```bash
cd group4
npm install
```

2. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start the server:**
```bash
npm start
# or for development:
npm run dev
```

4. **Test the API:**
```bash
curl http://localhost:3000/health
```

## 📡 API Endpoints

### 🎯 Main Navigation Request
```http
POST /navigate
Content-Type: application/json

{
  "query": "How do I get to Hongyadong without too many stairs?",
  "user_location": {
    "lat": 29.5647,
    "lng": 106.5507
  }
}
```

**Response:**
```json
{
  "success": true,
  "response": {
    "type": "navigation_response",
    "message": "Great choice! I found a nice route to Hongyadong that should work well for you. It's 1.2 km and should take about 15 minutes. Perfect news - this route is completely stair-free!",
    "route_summary": {
      "destination": "Hongyadong",
      "distance": "1.2 km", 
      "duration": "15 min",
      "accessibility_score": 92,
      "key_features": ["No stairs", "1 elevator(s)", "Excellent accessibility"]
    },
    "step_by_step": [...],
    "accessibility_info": {...},
    "quick_tips": [...]
  },
  "session_id": "nav_1234567890_abc123def"
}
```

### 💬 Follow-up Queries
```http
POST /follow-up
Content-Type: application/json

{
  "query": "Are there any elevators on this route?",
  "session_id": "nav_1234567890_abc123def"
}
```

### 🚶‍♀️ Live Navigation
```http
POST /start-live
Content-Type: application/json

{
  "session_id": "nav_1234567890_abc123def"
}
```

### 📍 Place Search
```http
POST /search-places
Content-Type: application/json

{
  "query": "restaurants near me",
  "user_location": {
    "lat": 29.5647,
    "lng": 106.5507
  },
  "filters": {
    "accessibility_required": true
  }
}
```

## 🧪 Example Usage

### Accessibility-Focused Navigation
```javascript
// User says: "How do I get to Hongyadong without too many stairs?"

// 1. AI interprets intent
const intent = {
  destination: "Hongyadong",
  accessibility_needs: ["avoid_stairs"],
  transportation_mode: "walking"
}

// 2. Maps API provides routes with elevation data
// 3. AI processes and scores routes for accessibility  
// 4. Natural response generated:

"Great choice! I found a nice route to Hongyadong that should work well for you. 
It's 1.2 km and should take about 15 minutes. Perfect news - this route is 
completely stair-free! There's an elevator available halfway through at the 
shopping center."
```

### Follow-up Conversation
```javascript
// User: "What if it's raining?"
// AI: "Good thinking! This route has covered walkways for about 70% of the journey. 
// There are 2 indoor sections you can use during the rain, and the elevator 
// I mentioned earlier will keep you dry for the vertical parts."
```

## 🏃‍♂️ Development

### Project Structure
- **`intentInterpreter.js`**: Uses Gemini AI to parse user requests and extract navigation intent
- **`mapsService.js`**: Interfaces with Google Maps APIs and analyzes routes for accessibility
- **`routeProcessor.js`**: Processes route data, calculates accessibility scores, generates recommendations
- **`responseGenerator.js`**: Creates natural language responses using Gemini AI
- **`navigationAssistant.js`**: Orchestrates the entire AI-Maps processing loop

### Key Classes

#### 🧠 IntentInterpreter
```javascript
const interpreter = new IntentInterpreter(apiKey);
const intent = await interpreter.interpretUserIntent(
  "Get me to the mall avoiding stairs"
);
// Returns: { destination, constraints, accessibility_needs, transportation_mode }
```

#### 🗺️ MapsService  
```javascript
const maps = new MapsService(apiKey);
const routes = await maps.getRoutes(origin, destination, intent);
// Returns enhanced routes with accessibility analysis
```

#### 🔄 RouteProcessor
```javascript
const processor = new RouteProcessor();
const processed = processor.processRoutes(routeData, userIntent);
// Returns routes sorted by accessibility score with user-friendly summaries
```

#### 💬 ResponseGenerator
```javascript
const generator = new ResponseGenerator(apiKey);
const response = await generator.generateNavigationResponse(
  processedRouteData, originalIntent, userLocation
);
// Returns natural language guidance with step-by-step instructions
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_MAPS_API_KEY=your_maps_api_key

# Optional  
PORT=3000
NODE_ENV=development
```

### Google Maps API Setup
Enable these APIs in Google Cloud Console:
1. **Directions API** - For route calculations
2. **Places API** - For location search
3. **Elevation API** - For accessibility analysis  
4. **Geocoding API** - For address resolution

## 📊 Accessibility Scoring

Routes are scored on a 0-100 scale based on:
- **Stairs present** (-20 points if user wants to avoid)
- **Steep grades** (-5 to -30 points based on severity)
- **Elevators available** (+10 points)
- **Ramps available** (+10 points)
- **General warnings** (-5 points each)

## 🧪 Testing

```bash
# Run tests
npm test

# Test the demo endpoint
curl http://localhost:3000/demo

# Test navigation request
curl -X POST http://localhost:3000/navigate \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I get to Times Square avoiding stairs?", "user_location": {"lat": 40.7589, "lng": -73.9851}}'
```

## 🤝 Contributing

Group 4 project implementing the AI + Google Maps navigation flow:
1. User intent interpretation with Gemini AI
2. Google Maps API integration  
3. Accessibility-aware route processing
4. Natural language response generation
5. Live navigation capabilities

## 📄 License

MIT License - Group 4 CQU Models Project