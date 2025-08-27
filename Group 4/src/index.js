import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { NavigationAssistant } from './navigationAssistant.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize Navigation Assistant
const navigationAssistant = new NavigationAssistant(
  process.env.GEMINI_API_KEY,
  process.env.GOOGLE_MAPS_API_KEY
);

// Routes

// Main navigation request endpoint
app.post('/navigate', async (req, res) => {
  try {
    const { query, user_location } = req.body;
    
    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }

    console.log('🚀 New navigation request:', query);
    
    const result = await navigationAssistant.processNavigationRequest(query, user_location);
    res.json(result);
    
  } catch (error) {
    console.error('Navigation endpoint error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Follow-up query endpoint
app.post('/follow-up', async (req, res) => {
  try {
    const { query, session_id } = req.body;
    
    if (!query || !session_id) {
      return res.status(400).json({
        success: false,
        error: 'Query and session_id are required'
      });
    }

    const result = await navigationAssistant.processFollowUpQuery(query, session_id);
    res.json({
      success: true,
      response: result
    });
    
  } catch (error) {
    console.error('Follow-up endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Start live navigation
app.post('/start-live', async (req, res) => {
  try {
    const { session_id } = req.body;
    
    if (!session_id) {
      return res.status(400).json({
        success: false,
        error: 'session_id is required'
      });
    }

    const result = await navigationAssistant.startLiveNavigation(session_id);
    res.json({
      success: true,
      navigation: result
    });
    
  } catch (error) {
    console.error('Live navigation start error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Update navigation progress
app.post('/update-progress', async (req, res) => {
  try {
    const { session_id, current_location } = req.body;
    
    if (!session_id) {
      return res.status(400).json({
        success: false,
        error: 'session_id is required'
      });
    }

    const result = await navigationAssistant.updateNavigationProgress(current_location, session_id);
    res.json({
      success: true,
      update: result
    });
    
  } catch (error) {
    console.error('Progress update error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Search nearby places
app.post('/search-places', async (req, res) => {
  try {
    const { query, user_location, filters } = req.body;
    
    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }

    const result = await navigationAssistant.searchNearbyPlaces(query, user_location, filters || {});
    res.json(result);
    
  } catch (error) {
    console.error('Places search error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'AI Navigation Assistant',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// Demo endpoint for testing
app.get('/demo', (req, res) => {
  res.json({
    message: 'AI Navigation Assistant Demo',
    example_requests: {
      navigate: {
        method: 'POST',
        endpoint: '/navigate',
        body: {
          query: "How do I get to Hongyadong without too many stairs?",
          user_location: { lat: 29.5647, lng: 106.5507 }
        }
      },
      follow_up: {
        method: 'POST',
        endpoint: '/follow-up', 
        body: {
          query: "Are there any elevators on this route?",
          session_id: "nav_1234567890_abc123def"
        }
      }
    },
    features: [
      "🤖 AI-powered intent interpretation",
      "🗺️ Google Maps integration with accessibility analysis", 
      "♿ Accessibility-aware route processing",
      "💬 Natural language responses",
      "🚶‍♀️ Live navigation guidance",
      "🔄 Follow-up query handling"
    ]
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    available_endpoints: ['/navigate', '/follow-up', '/start-live', '/update-progress', '/search-places', '/health', '/demo']
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
🚀 AI Navigation Assistant Server Started!

📍 Server running on port ${PORT}
🌐 Health check: http://localhost:${PORT}/health  
🎯 Demo info: http://localhost:${PORT}/demo

🔧 Environment:
   - Node.js: ${process.version}
   - Gemini API: ${process.env.GEMINI_API_KEY ? '✅ Configured' : '❌ Missing'}
   - Google Maps API: ${process.env.GOOGLE_MAPS_API_KEY ? '✅ Configured' : '❌ Missing'}

📚 Ready to process navigation requests with AI-powered accessibility features!
  `);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 SIGTERM received, shutting down gracefully');
  navigationAssistant.clearSession();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('🛑 SIGINT received, shutting down gracefully');  
  navigationAssistant.clearSession();
  process.exit(0);
});

export default app;