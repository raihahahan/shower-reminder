import requests
import base64

def validate_photo(photo):
    try:
        base64_photo = base64.b64encode(photo).decode('utf-8')
        
        # Make request to Flask API
        response = requests.post(
            "http://your-api-url:5000/predict",
            json={"image": base64_photo}
        )
        
        if response.status_code == 200:
            result = response.json()
            is_showerhead = result["is_showerhead"]
            confidence = result["confidence"]
            
            # Add confidence threshold
            if confidence < 0.5:  # Adjust threshold as needed
                return False
                
            return is_showerhead
            
    except Exception as e:
        print(f"Error in validation: {e}")
        return False
    
    return False