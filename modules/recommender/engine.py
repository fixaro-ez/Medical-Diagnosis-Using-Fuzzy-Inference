import pickle
import os
import sys

class SymptomRecommender:
    def __init__(self):
        # Locate the pkl file relative to this script
        self.model_path = os.path.join(os.path.dirname(__file__), 'sympton_model.pkl')
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            print(f"\n[Error] Model file not found at: {self.model_path}")
            print("Please ensure 'sympton_model.pkl' is in the 'modules/recommender' folder.")
            self.model = None
        except Exception as e:
            print(f"\n[Error] Failed to load model: {e}")
            self.model = None


    def predict(self, user_text):
        if not self.model:
            return None
        
        try:
            # Get probabilities for ALL classes
            # classes_ stores the labels ['diabetes', 'heart', etc.]
            class_labels = self.model.classes_ 
            probs = self.model.predict_proba([user_text])[0]
            
            # Zip them together: [('diabetes', 0.1), ('heart', 0.8)...]
            results = list(zip(class_labels, probs))
            
            # Sort by probability (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 3 for the doctor to review
            return results[:3] 
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

if __name__ == "__main__":
    rec = SymptomRecommender()
    # Test with ambiguous input
    top_3 = rec.predict("I have chest pain but also a fever")
    
    print("\n--- Differential Diagnosis Report ---")
    for disease, probability in top_3:
        print(f"{disease.upper()}: {probability*100:.2f}%")