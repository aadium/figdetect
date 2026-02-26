import json
import pandas as pd
import re
from tqdm import tqdm

def create_mega_dataframe(instances_path, figures_path, sources_path):
    with open(instances_path, 'r', encoding='utf-8') as f:
        instances = json.load(f)
    with open(figures_path, 'r', encoding='utf-8') as f:
        figures = json.load(f)
    with open(sources_path, 'r', encoding='utf-8') as f:
        sources = json.load(f)

    fig_map = {f['id']: f for f in figures}
    src_map = {s['id']: s for s in sources}

    rows = []

    for inst in tqdm(instances, desc="Processing Instances"):
        text = inst.get('text', '')
        text = text.replace("&nbsp;", " ").replace('\ufeff', '')
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        source_id = inst.get('source', {}).get('id')
        source_data = src_map.get(source_id, {}).get('cache', {})

        creators = source_data.get('creators', [])
        formatted_authors = []
        for c in creators:
            if 'firstName' in c and 'lastName' in c:
                formatted_authors.append(f"{c['lastName']}, {c['firstName']}")
            elif 'name' in c:
                formatted_authors.append(c['name'])
        all_authors = "; ".join(formatted_authors) if formatted_authors else "Unknown"

        for anno in inst.get('annotations', []):
            fig_id = anno.get('figure', {}).get('id')
            fig_info = fig_map.get(fig_id, {})
            current_anno_id = anno.get('id')

            label_groups = {}
            for hl in anno.get('highlights', []):
                label_val = hl.get('label', 0) 
                start, end = hl.get('start', 0), hl.get('end', 0)
                snippet = clean_text[start:end]
                
                label_groups.setdefault(label_val, []).append(snippet)

            # Creating one row per label group within the annotation
            if fig_info.get('allow_in_gofigure', True):
                for label_key, snippets in label_groups.items():
                    rows.append({
                        "highlights": "; ".join(snippets),
                        "figure_name": fig_info.get('name'),
                        "figure_type": fig_info.get('type', {}).get('name') if isinstance(fig_info.get('type'), dict) else "N/A",
                        "authors": all_authors,
                        "source_title": source_data.get('title'),
                        "source_year": source_data.get('date'),
                        "full_text": clean_text,
                        "instance_id": inst.get('id'),
                        "annotation_id": current_anno_id,
                        "ref": inst.get('ref')
                    })

    return pd.DataFrame(rows)

df = create_mega_dataframe('./input/instances.json', './input/figures.json', './input/sources.json')
df.to_csv('./training/rhetorical_analysis_export_gofigure.csv', index=False)