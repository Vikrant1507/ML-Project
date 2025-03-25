# Vomitoxin Prediction Flask Application

## Setup and Installation

1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Ensure you have the `model.pkl` file in the same directory

## Running the Application

### Development Mode
```bash
flask run
```

## Endpoints

- `/`: Home page with prediction input form
- `/predict`: Prediction endpoint (supports both web form and JSON)
- `/health`: Health check endpoint

## API Usage

### Web Interface
- Navigate to the home page
- Fill in the feature inputs
- Click "Predict Vomitoxin Level"

### JSON API
```python
import requests

# Example prediction request
response = requests.post('http://localhost:5000/predict', 
                         json={'features': [0.1, 0.2, 0.3, ...]})
prediction = response.json()['prediction']
```
