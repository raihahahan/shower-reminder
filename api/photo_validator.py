import requests
import base64

def validate_photo(photo):
    try:
        base64_photo = base64.b64encode(photo).decode('utf-8')
        
        # Make request to Flask API
        response = requests.post(
            "http://localhost:8000//predict",
            json={"image": base64_photo}
        )
        
        result = response.json()
        is_showerhead = result.get("is_showerhead", False)
        confidence = result.get("confidence", 0.0)
        predicted_class = result.get("predicted_class", "Unknown")
        is_confident = result.get("is_confident", False)
        ai_label = result.get("ai_label", "Unknown")  # Retrieve AI-predicted label
        
        # Return the `is_showerhead` value if confidence is sufficient
        return is_showerhead, confidence, predicted_class, is_confident, ai_label
            
    except Exception as e:
        print(f"Error in validation: {e}")
        return False