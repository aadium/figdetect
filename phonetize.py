import pandas as pd
from phonemizer import phonemize
from phonemizer.separator import Separator

def generate_phonetic_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df['full_text'] = df['full_text'].fillna("")
    df['highlights'] = df['highlights'].fillna("")

    separator = Separator(phone=" ", word=" <W> ")
    
    # 1. Phonemize Full Text (Safe bulk operation)
    print("Transcribing full_text...")
    df['full_text_phonetic'] = phonemize(
        df['full_text'].tolist(),
        language='en-us', backend='espeak', separator=separator,
        strip=True, njobs=5,
        preserve_empty_lines=True
    )
    
    # 2. Optimized Highlights Phonemization
    print("Preparing highlights for bulk transcription...")
    all_parts = []
    row_map = []

    for i, entry in enumerate(df['highlights'].tolist()):
        parts = entry.split(';')
        for p in parts:
            all_parts.append(p)
            row_map.append(i)

    print(f"Transcribing {len(all_parts)} highlight fragments in bulk...")
    phonemized_all = phonemize(
        all_parts,
        language='en-us', backend='espeak', separator=separator,
        strip=True, njobs=5, preserve_empty_lines=True
    )

    # 3. Reconstruct the rows with semicolons
    print("Reconstructing rows...")
    reconstructed = [[] for _ in range(len(df))]
    for ph_part, original_row_idx in zip(phonemized_all, row_map):
        reconstructed[original_row_idx].append(ph_part)

    df['highlights_phonetic'] = ["; ".join(r) for r in reconstructed]

    df.to_csv(output_file, index=False)
    print(f"Success! Saved to {output_file}")

if __name__ == "__main__":
    generate_phonetic_csv('./training/rhetorical_analysis_english_gofigure.csv', './training/gofigure_phonetized.csv')