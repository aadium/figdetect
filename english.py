import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm

DetectorFactory.seed = 42

def extract_english_instances(input_path, output_path):
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    memo = {}
    english_indices = []

    print("Detecting languages...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['full_text'])
        
        if text in memo:
            is_en = memo[text]
        else:
            try:
                is_en = detect(text) == 'en'
            except:
                is_en = False
            memo[text] = is_en
        
        if is_en:
            english_indices.append(idx)

    df_english = df.iloc[english_indices].copy()
    df_english.to_csv(output_path, index=False)
    
    print(f"\nExtraction Complete!")
    print(f"Original instances: {len(df)}")
    print(f"English instances:  {len(df_english)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    extract_english_instances(
        "./training/rhetorical_analysis_export_gofigure.csv", 
        "./training/rhetorical_analysis_english_gofigure.csv"
    )