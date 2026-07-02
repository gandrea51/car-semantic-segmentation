from ollama import Client
import numpy as np

client = Client(host="http://localhost:11434")

CLASSES = {0: "background", 1: "body", 2: "wheel", 3: "window", 4: "light"}
LLM = "qwen3:4b"

def build_report(pred_mask, conf_map, entropy_map):
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    distribution = {}

    for c, count in zip(unique, counts):
        distribution[CLASSES.get(int(c), str(c))] = round(count / total_pixels * 100, 2)
    
    high_entropy_tresh = np.percentile(entropy_map, 95)
    hotspots = int(np.sum(entropy_map > high_entropy_tresh))
    
    report = {
        "class_distribution": distribution,
        "mean_confidence": round(float(conf_map.mean()), 3),
        "min_confidence": round(float(conf_map.min()), 3),
        "mean_entropy": round(float(entropy_map.mean()), 3),
        "max_entropy": round(float(entropy_map.max()), 3),
        "high_entropy_pixels": hotspots
    }
    return report

def generate_explanation(report, model_name):
    prompt = f"""
        You are a Computer Vision expert.
        Semantic segmentation report:
            Model: {model_name}
            Statistics: {report}
            Explain:
                1. What objects/components are dominant.
                2. Confidence quality.
                3. Uncertainty behaviour.
                4. Possibile waek regions.
        Use simple English (Maximum 120 words).
    """

    response = client.chat(model=LLM, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


