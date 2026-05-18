import pandas as pd
import json
import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
llm = importlib.import_module("02_llm_skill_extraction")
from google import genai
from google.genai import types

def test():
    llm.init_model(llm.PROJECT, llm.LOCATION)
    df = pd.read_csv("calibration_dataset.csv")
    req = df.iloc[2]["requirement"]
    
    prompt = llm.build_few_shot_prompt(req)
    
    config = {
        "temperature": llm.TEMPERATURE,
        "max_output_tokens": llm.MAX_TOKENS,
        "system_instruction": llm.SYSTEM_PROMPT,
        # "response_mime_type": "application/json"
    }

    print("Calling API...")
    response = llm.CLIENT.models.generate_content(
        model=llm.MODEL_NAME,
        contents=prompt,
        config=config
    )
    
    raw_text = getattr(response, "text", None)
    
    print("\n--- RAW TEXT ---")
    print(raw_text)
    if response.candidates:
        print(f"Finish reason: {response.candidates[0].finish_reason}")
    
    with open("debug_raw_output.json", "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    print("\nAttempting to parse JSON...")
    try:
        json.loads(raw_text)
        print("Parse successful!")
    except Exception as e:
        print(f"Parse error: {e}")

if __name__ == "__main__":
    test()
