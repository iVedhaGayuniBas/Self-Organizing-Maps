import pickle
import os

def save_contexts(som_contexts_scores, cosine_contexts_scores):
    """
    Save context data to 'contexts' directory
    """
    # Create contexts directory
    os.makedirs("contexts", exist_ok=True)
    
    # Save as pickle files
    with open("contexts/som_contexts_scores.pkl", 'wb') as f:
        pickle.dump(som_contexts_scores, f)
    
    with open("contexts/cosine_contexts_scores.pkl", 'wb') as f:
        pickle.dump(cosine_contexts_scores, f)
    
    print("✅ Contexts saved to 'contexts' directory")

def load_contexts():
    """
    Load context data from 'contexts' directory
    """
    with open("contexts/som_contexts_scores.pkl", 'rb') as f:
        som_contexts_scores = pickle.load(f)
    
    with open("contexts/cosine_contexts_scores.pkl", 'rb') as f:
        cosine_contexts_scores = pickle.load(f)
    
    return som_contexts_scores, cosine_contexts_scores 