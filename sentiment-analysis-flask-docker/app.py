# app.py
from flask import Flask, request, jsonify, render_template_string
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

# Download NLTK data if not already present
try:
    # Attempt to download commonly required NLTK data quietly
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception:
    # If automatic download fails (e.g., no network in container), provide a clear message
    print("Warning: could not auto-download NLTK data. If you see tokenizer errors,")
    print("run: python3 -m nltk.downloader punkt stopwords\nor rebuild the Docker image which pre-downloads NLTK data.")

# Initialize Flask app
app = Flask(__name__)

# ========================
# Load Model and Vectorizer
# ========================
print("Loading model and vectorizer...")

try:
    # Load trained model
    with open('models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load TF-IDF vectorizer
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✓ Model and vectorizer loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to train and save the model.")
    exit(1)

# Define stopwords and tokenizer
stop_words = set(stopwords.words('english'))
_word_tokenizer = TreebankWordTokenizer()

# ========================
# Text Preprocessing Function
# ========================
def preprocess_text(text):
    """
    Preprocess input text (same as training preprocessing)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization (use Treebank tokenizer to avoid requiring punkt)
    tokens = _word_tokenizer.tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# ========================
# API Routes
# ========================

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint - API information
    """
    return jsonify({
        'message': 'Sentiment Analysis API',
        'version': '1.0',
        'endpoints': {
            '/': 'GET - API information',
            '/predict': 'POST - Predict sentiment of text',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects JSON: {"text": "your review text here"}
    Returns: {"sentiment": "positive/negative", "confidence": 0.95}
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        if 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in JSON'
            }), 400
        
        # Get text from request
        text = data['text']
        
        # Validate text is not empty
        if not text or text.strip() == '':
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Preprocess the text
        cleaned_text = preprocess_text(text)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = float(max(probabilities))
        
        # Map prediction to sentiment
        sentiment = 'positive' if prediction == 1 else 'negative'
        
        # Prepare response
        response = {
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'negative': round(float(probabilities[0]), 4),
                'positive': round(float(probabilities[1]), 4)
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500
# Add this import at the top with other imports
from flask import Flask, request, jsonify, render_template_string

# ... (keep all your existing code - model loading, preprocessing function, etc.)

# Add these new routes after your existing /predict route:

@app.route('/test', methods=['GET'])
def test_page():
    """
    Professional sentiment analysis dashboard
    """
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis API - Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                --primary: #0ea5e9;
                --primary-dark: #0284c7;
                --secondary: #38bdf8;
                --success: #10b981;
                --danger: #f43f5e;
                --warning: #fbbf24;
                --dark: #020617;
                --dark-lighter: #0f172a;
                --dark-card: #0f172a;
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-muted: #94a3b8;
                --border: #1e293b;
                --glass-bg: rgba(15, 23, 42, 0.8);
                --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
            }

                        body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #020617;
                color: var(--text-primary);
                min-height: 100vh;
                padding: 20px;
                position: relative;
                overflow-x: hidden;
            }

            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-image: 
                    radial-gradient(at 20% 30%, rgba(14, 165, 233, 0.15) 0px, transparent 50%),
                    radial-gradient(at 80% 70%, rgba(56, 189, 248, 0.15) 0px, transparent 50%);
                pointer-events: none;
                z-index: -1;
            }
  
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            /* Header */
            .header {
                margin-top:100px;
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 24px 32px;
                margin-bottom: 24px;
                box-shadow: var(--shadow);
            }
            
            .header-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 16px;
            }
            
            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .brand-icon {
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
            }
            
            .brand-text h1 {
                font-size: 24px;
                font-weight: 700;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .brand-text p {
                font-size: 14px;
                color: var(--text-muted);
                margin-top: 2px;
            }
            
            .status-badge {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 8px;
                font-size: 14px;
                color: var(--success);
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                background: var(--success);
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            /* Main Grid */
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 24px;
            }
            
            @media (max-width: 968px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
            
            /* Cards */
            .card {
                margin-top: 10px;
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 28px;
                box-shadow: var(--shadow);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 25px 30px -5px rgba(0, 0, 0, 0.4);
            }
            
            .card-header {
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 20px;
            }
            
            .card-icon {
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            }
            
            .card-title {
                font-size: 18px;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            /* Input Section */
            .full-width {
                grid-column: 1 / -1;
            }
            
            label {
                display: block;
                font-size: 14px;
                font-weight: 500;
                color: var(--text-secondary);
                margin-bottom: 12px;
            }
            
            textarea {
                width: 100%;
                padding: 16px;
                background: var(--dark-lighter);
                border: 2px solid var(--border);
                border-radius: 12px;
                font-size: 15px;
                font-family: inherit;
                color: var(--text-primary);
                resize: vertical;
                min-height: 140px;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            
            textarea:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }
            
            textarea::placeholder {
                color: var(--text-muted);
            }
            
            .button-group {
                display: flex;
                gap: 12px;
                margin-top: 16px;
            }
            
            button {
                flex: 1;
                padding: 14px 24px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
            }
            
            button:active:not(:disabled) {
                transform: translateY(0);
            }
            
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .btn-secondary {
                background: var(--dark-lighter);
                border: 1px solid var(--border);
            }
            
            .btn-secondary:hover:not(:disabled) {
                background: var(--dark);
                box-shadow: none;
            }
            
            /* Loading State */
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid var(--border);
                border-top-color: var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 12px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* Result Cards */
            .result-card {
                display: none;
                animation: slideUp 0.4s ease;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .sentiment-display {
                text-align: center;
                padding: 32px;
                background: var(--dark-lighter);
                border-radius: 12px;
                margin-bottom: 20px;
            }
            
            .sentiment-emoji {
                font-size: 64px;
                margin-bottom: 16px;
                display: block;
            }
            
            .sentiment-label {
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 8px;
            }
            
            .sentiment-label.positive {
                color: var(--success);
            }
            
            .sentiment-label.negative {
                color: var(--danger);
            }
            
            .confidence-score {
                font-size: 16px;
                color: var(--text-muted);
            }
            
            .confidence-number {
                font-size: 20px;
                font-weight: 600;
                color: var(--text-secondary);
            }
            
            /* Probability Bars */
            .probability-section h3 {
                font-size: 16px;
                color: var(--text-secondary);
                margin-bottom: 16px;
            }
            
            .prob-item {
                margin-bottom: 16px;
            }
            
            .prob-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .prob-name {
                font-size: 14px;
                font-weight: 500;
                color: var(--text-secondary);
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .prob-value {
                font-size: 16px;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .prob-bar-bg {
                height: 8px;
                background: var(--dark-lighter);
                border-radius: 4px;
                overflow: hidden;
            }
            
            .prob-bar-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 0.8s ease;
            }
            
            .prob-bar-fill.positive {
                background: linear-gradient(90deg, var(--success), #34d399);
            }
            
            .prob-bar-fill.negative {
                background: linear-gradient(90deg, var(--danger), #f87171);
            }
            
            /* Stats Cards */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 12px;
            }
            
            .stat-box {
                background: var(--dark-lighter);
                padding: 16px;
                border-radius: 10px;
                border: 1px solid var(--border);
            }
            
            .stat-label {
                font-size: 12px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }
            
            .stat-value {
                font-size: 20px;
                font-weight: 700;
                color: var(--text-primary);
            }
            
            /* Error State */
            .error-card {
                display: none;
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 12px;
                padding: 20px;
                margin-top: 16px;
            }
            
            .error-content {
                display: flex;
                align-items: start;
                gap: 12px;
            }
            
            .error-icon {
                font-size: 24px;
            }
            
            .error-text {
                flex: 1;
            }
            
            .error-title {
                font-size: 16px;
                font-weight: 600;
                color: var(--danger);
                margin-bottom: 4px;
            }
            
            .error-message {
                font-size: 14px;
                color: var(--text-muted);
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <!-- Header -->
            <div class="header">
                <div class="header-content">
                    <div class="brand">
                        <div class="brand-icon">🎭</div>
                        <div class="brand-text">
                            <h1>Sentiment Analysis API</h1>
                            <p>Advanced NLP-powered sentiment detection</p>
                        </div>
                    </div>
                    <div class="status-badge">
                        <span class="status-dot"></span>
                        <span>System Online</span>
                    </div>
                </div>
            </div>
            
            <!-- Main Grid -->
            <div class="grid">
                <!-- Input Card -->
                <div class="card full-width">
                    <div class="card-header">
                        <div class="card-icon">✍️</div>
                        <h2 class="card-title">Text Input</h2>
                    </div>
                    
                    <form id="analysisForm">
                        <label for="textInput">Enter text to analyze sentiment</label>
                        <textarea 
                            id="textInput" 
                            name="text" 
                            placeholder="Type or paste any text here... Movie reviews, product feedback, customer comments, social media posts, etc."
                            required
                        ></textarea>
                        
                        <div class="button-group">
                            <button type="submit" id="analyzeBtn">
                                <span>🔍</span>
                                <span>Analyze Sentiment</span>
                            </button>
                            <button type="button" class="btn-secondary" onclick="clearForm()">
                                <span>🗑️</span>
                                <span>Clear</span>
                            </button>
                        </div>
                    </form>
                    
                    <div class="error-card" id="errorCard">
                        <div class="error-content">
                            <div class="error-icon">⚠️</div>
                            <div class="error-text">
                                <div class="error-title">Error</div>
                                <div class="error-message" id="errorMessage"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Loading State -->
                <div class="card full-width loading" id="loadingCard">
                    <div class="spinner"></div>
                    <p style="color: var(--text-secondary); font-weight: 500;">Analyzing text with ML model...</p>
                </div>
                
                <!-- Result: Sentiment Display -->
                <div class="card result-card" id="sentimentCard">
                    <div class="card-header">
                        <div class="card-icon">📊</div>
                        <h2 class="card-title">Sentiment Result</h2>
                    </div>
                    
                    <div class="sentiment-display">
                        <span class="sentiment-emoji" id="sentimentEmoji"></span>
                        <div class="sentiment-label" id="sentimentLabel"></div>
                        <div class="confidence-score">
                            Confidence: <span class="confidence-number" id="confidenceValue"></span>
                        </div>
                    </div>
                </div>
                
                <!-- Result: Probability Breakdown -->
                <div class="card result-card" id="probabilityCard">
                    <div class="card-header">
                        <div class="card-icon">📈</div>
                        <h2 class="card-title">Probability Distribution</h2>
                    </div>
                    
                    <div class="probability-section">
                        <h3>Model Confidence Scores</h3>
                        
                        <div class="prob-item">
                            <div class="prob-header">
                                <span class="prob-name">
                                    <span>😊</span>
                                    <span>Positive</span>
                                </span>
                                <span class="prob-value" id="posValue">0%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill positive" id="posBar" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <div class="prob-item">
                            <div class="prob-header">
                                <span class="prob-name">
                                    <span>😞</span>
                                    <span>Negative</span>
                                </span>
                                <span class="prob-value" id="negValue">0%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill negative" id="negBar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Result: Statistics -->
                <div class="card result-card full-width" id="statsCard">
                    <div class="card-header">
                        <div class="card-icon">📋</div>
                        <h2 class="card-title">Analysis Details</h2>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Text Length</div>
                            <div class="stat-value" id="textLength">-</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Word Count</div>
                            <div class="stat-value" id="wordCount">-</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Model</div>
                            <div class="stat-value" style="font-size: 14px;">TF-IDF + LR</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Processing Time</div>
                            <div class="stat-value" id="processingTime">-</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const form = document.getElementById('analysisForm');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loadingCard = document.getElementById('loadingCard');
            const errorCard = document.getElementById('errorCard');
            const resultCards = document.querySelectorAll('.result-card');
            
            let startTime;
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const text = document.getElementById('textInput').value.trim();
                
                if (!text) {
                    showError('Please enter some text to analyze');
                    return;
                }
                
                // Hide previous results
                hideAllResults();
                loadingCard.style.display = 'block';
                analyzeBtn.disabled = true;
                startTime = Date.now();
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResults(data, text);
                    } else {
                        showError(data.error || 'An error occurred during analysis');
                    }
                } catch (error) {
                    showError('Failed to connect to the API. Please ensure the server is running.');
                } finally {
                    loadingCard.style.display = 'none';
                    analyzeBtn.disabled = false;
                }
            });
            
            function displayResults(data, originalText) {
                const sentiment = data.sentiment;
                const confidence = (data.confidence * 100).toFixed(2);
                const posProb = (data.probabilities.positive * 100).toFixed(2);
                const negProb = (data.probabilities.negative * 100).toFixed(2);
                const processingTime = Date.now() - startTime;
                
                // Sentiment display
                document.getElementById('sentimentEmoji').textContent = sentiment === 'positive' ? '😊' : '😞';
                document.getElementById('sentimentLabel').textContent = sentiment.toUpperCase();
                document.getElementById('sentimentLabel').className = `sentiment-label ${sentiment}`;
                document.getElementById('confidenceValue').textContent = confidence + '%';
                
                // Probability bars
                document.getElementById('posValue').textContent = posProb + '%';
                document.getElementById('posBar').style.width = posProb + '%';
                document.getElementById('negValue').textContent = negProb + '%';
                document.getElementById('negBar').style.width = negProb + '%';
                
                // Statistics
                document.getElementById('textLength').textContent = originalText.length;
                document.getElementById('wordCount').textContent = originalText.split(/\s+/).length;
                document.getElementById('processingTime').textContent = processingTime + 'ms';
                
                // Show all result cards
                resultCards.forEach(card => {
                    card.style.display = 'block';
                });
                
                // Scroll to results
                setTimeout(() => {
                    document.getElementById('sentimentCard').scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'nearest' 
                    });
                }, 100);
            }
            
            function showError(message) {
                document.getElementById('errorMessage').textContent = message;
                errorCard.style.display = 'block';
                setTimeout(() => {
                    errorCard.style.display = 'none';
                }, 5000);
            }
            
            function hideAllResults() {
                errorCard.style.display = 'none';
                resultCards.forEach(card => {
                    card.style.display = 'none';
                });
            }
            
            function clearForm() {
                document.getElementById('textInput').value = '';
                hideAllResults();
                document.getElementById('textInput').focus();
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)


# Keep your existing code below (the if __name__ == '__main__': section)

# ========================
# Run the App
# ========================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Sentiment Analysis API...")
    print("API will be available at: http://0.0.0.0:5000")
    print("="*50 + "\n")
    print("Starting Sentiment Analysis Web App...")
    print("Web App will be available at: http://0.0.0.0:5000/test")
    print("="*50 + "\n")
    
    app.run(
        host='0.0.0.0',  # Makes server accessible externally
        port=5000,
        debug=True  # Enable debug mode for development
    )
