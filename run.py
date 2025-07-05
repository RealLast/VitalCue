import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fastapi import FastAPI, Request, HTTPException
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "EmbedHealth")))

from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from prompt.prompt_with_answer import PromptWithAnswer

import logging

app = FastAPI()

# Global model variable
model = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
def load_model():
    """Load the trained EmbedHealthFlamingo model at startup."""
    global model
    
    try:
        # Initialize the model with the same configuration as training
        # Handle device selection more carefully to avoid MPS issues
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            try:
                # Test MPS availability
                test_tensor = torch.tensor([1.0], device="mps")
                device = "mps"
            except:
                print("⚠️  MPS device available but has issues, falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
        
        print(f"🔧 Using device: {device}")
        
        # Initialize model with CPU first to avoid MPS loading issues
        model = EmbedHealthFlamingo(
            device="cpu",  # Start with CPU to avoid MPS loading issues
            llm_id="google/gemma-2b",  # Updated to use gemma-2b
            cross_attn_every_n_layers=1,
        )
        
        # Load the trained checkpoint
        checkpoint_path = "/Users/planger/Development/Hacking4Health/results/EmbedHealthFlamingo/stage2_captioning/checkpoints/best_model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint with CPU to avoid MPS issues
        model.load_from_file(checkpoint_path)
        
        # Move model to target device after loading
        if device != "cpu":
            model = model.to(device)
            model.device = device
        
        model.eval()  # Set to evaluation mode
        
        print(f"✅ Model loaded successfully from {checkpoint_path}")
        print(f"📱 Device: {model.device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def validate_time_series_data(time_series_data: List[float]) -> torch.Tensor:
    """Validate and convert time series data to tensor."""
    if not isinstance(time_series_data, list):
        raise ValueError("Time series data must be a list of numbers")
    
    if len(time_series_data) == 0:
        raise ValueError("Time series data cannot be empty")
    
    # Convert to numpy array and then to tensor
    try:
        ts_array = np.array(time_series_data, dtype=np.float32)
        ts_tensor = torch.tensor(ts_array, dtype=torch.float32)
        
        # Ensure it's 2D: (1, sequence_length)
        if ts_tensor.ndim == 1:
            ts_tensor = ts_tensor.unsqueeze(0)
        
        return ts_tensor
        
    except Exception as e:
        raise ValueError(f"Invalid time series data: {e}")

def create_text_time_series_prompt_with_stats(text: str, time_series_data: List[float]) -> TextTimeSeriesPrompt:
    """Create a TextTimeSeriesPrompt with mean and std included in the text."""
    ts_array = np.array(time_series_data, dtype=np.float32)
    mean_val = np.mean(ts_array)
    std_val = np.std(ts_array)

    # Include statistics in the text
    enhanced_text = f"{text} It has mean {mean_val:.4f} and std {std_val:.4f}."

    return TextTimeSeriesPrompt(enhanced_text, ts_array)


def create_batch_item(
    pre_prompt: str,
    time_series_text: List[str],
    time_series_data: List[List[float]],
    post_prompt: str = ""
) -> Dict[str, Any]:
    """Create a batch item using PromptWithAnswer and its to_dict() method."""
    
    # Create pre-prompt
    pre_prompt_obj = TextPrompt(pre_prompt.strip()) if pre_prompt.strip() else TextPrompt("")
    
    # Create time series prompts with statistics
    text_time_series_prompts = []
    for text, ts_data in zip(time_series_text, time_series_data):
        ts_prompt = create_text_time_series_prompt_with_stats(text, np.array(ts_data))
        text_time_series_prompts.append(ts_prompt)
    
    # Create post-prompt
    post_prompt_obj = TextPrompt(post_prompt.strip()) if post_prompt.strip() else TextPrompt("")
    
    print("PROMPTS")
    print(text_time_series_prompts)
    # Create PromptWithAnswer and get the dictionary
    prompt_with_answer = PromptWithAnswer(
        pre_prompt=pre_prompt_obj.get_text(),
        text_time_series_prompt_list=text_time_series_prompts,
        post_prompt=post_prompt_obj.get_text(),
        answer=""  # Empty for generation
    )
    
    batch_dict = prompt_with_answer.to_dict()
    
    # Convert numpy arrays to torch tensors and concatenate into single tensor
    time_series_tensors = []
    for ts in batch_dict["time_series"]:
        # Convert numpy array to torch tensor
        ts_tensor = torch.tensor(ts, dtype=torch.float32)
        # Ensure 2D shape: (1, sequence_length)
        if ts_tensor.ndim == 1:
            ts_tensor = ts_tensor.unsqueeze(0)
        time_series_tensors.append(ts_tensor)
    
    # Concatenate all tensors along the first dimension to create a single tensor
    if time_series_tensors:
        combined_tensor = torch.cat(time_series_tensors, dim=0)
        
        # Normalize the time series (same as in training)
        means = combined_tensor.mean(dim=1, keepdim=True)
        stds = combined_tensor.std(dim=1, keepdim=True)
        combined_tensor = (combined_tensor - means) / (stds + 1e-8)
        
        # Pad to make sequence length divisible by patch_size (4)
        patch_size = 4
        current_length = combined_tensor.shape[1]
        padded_length = ((current_length + patch_size - 1) // patch_size) * patch_size
        
        if current_length < padded_length:
            # Pad with zeros
            padding = torch.zeros(combined_tensor.shape[0], padded_length - current_length, 
                               dtype=combined_tensor.dtype, device=combined_tensor.device)
            combined_tensor = torch.cat([combined_tensor, padding], dim=1)
        elif current_length > padded_length:
            # Truncate if longer
            combined_tensor = combined_tensor[:, :padded_length]
    else:
        # If no tensors, create an empty tensor
        combined_tensor = torch.empty((0, 0), dtype=torch.float32)
    
    # Update the dictionary with the single combined tensor
    batch_dict["time_series"] = combined_tensor
    
    return batch_dict

@app.post("/predict")
async def predict(request: Request):
    """
    Generate predictions using the trained EmbedHealthFlamingo model.
    
    Expected request format:
    {
        "pre_prompt": "string",  # Text before time series
        "time_series_text": ["string", ...],  # Text descriptions for each time series
        "time_series_data": [[float, ...], ...],  # Actual time series data
        "post_prompt": "string",  # Text after time series (optional)
        "max_new_tokens": 100  # Maximum tokens to generate (optional)
    }
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = ["pre_prompt", "time_series_text", "time_series_data"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required field: {field}"
                )
        
        
        # Extract data
        pre_prompt = data["pre_prompt"]
        time_series_text = data["time_series_text"]
        time_series_data = data["time_series_data"]
        post_prompt = data.get("post_prompt", "")
        max_new_tokens = data.get("max_new_tokens", 500)
        
        # Validate that we have matching text and data
        if len(time_series_text) != len(time_series_data):
            raise HTTPException(
                status_code=400,
                detail=f"Number of time series texts ({len(time_series_text)}) must match number of time series data ({len(time_series_data)})"
            )
        
        # Create batch item using PromptWithAnswer
        batch_item = create_batch_item(
            pre_prompt=pre_prompt,
            time_series_text=time_series_text,
            time_series_data=time_series_data,
            post_prompt=post_prompt
        )
        print(batch_item)
        
        # Generate prediction
        with torch.no_grad():
            predictions = model.generate(
                batch=[batch_item],
                max_new_tokens=max_new_tokens
            )
        
        # Return the generated text
        generated_text = predictions[0] if predictions else ""
        
        return {
            "generated_text": generated_text,
            "input_info": {
                "pre_prompt": pre_prompt,
                "time_series_count": len(time_series_data),
                "post_prompt": post_prompt,
                "max_new_tokens": max_new_tokens
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": model.device if model else None
    }

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "EmbedHealthFlamingo",
        "device": model.device,
        "llm_id": "google/gemma-2b",
        "cross_attn_every_n_layers": 1
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
