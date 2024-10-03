import pandas as pd
import numpy as np
import re
import os
import json
import PyPDF2
from PyPDF2 import PdfReader

class PaperProcessor:
    def __init__(self, metadata_file, keywords, data_folder, output_folder):
        self.metadata = pd.read_csv(metadata_file)
        self.keywords = keywords
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.papers = {}
        self.keyword_densities = {}
        
        # Create output directories if they don't exist
        self.high_relevant_dir = os.path.join(self.output_folder, 'high_relevant_data')
        self.relevant_dir = os.path.join(self.output_folder, 'relevant_data')
        os.makedirs(self.high_relevant_dir, exist_ok=True)
        os.makedirs(self.relevant_dir, exist_ok=True)
    
    def filter_papers(self):
        # Filter papers based on language
        english_papers_df = self.metadata[self.metadata['language'] == 'en']
        english_papers_df['title_lower'] = english_papers_df['title'].str.lower()
        english_papers_df['keyword_count'] = english_papers_df['title_lower'].apply(self.count_keywords_in_title)
        papers_to_load = english_papers_df[
            ((english_papers_df['relevance_score'] > 2.0) | (english_papers_df['keyword_count'] > 1))
        ]
        return papers_to_load

    def count_keywords_in_title(self, title):
        count = sum(1 for keyword in self.keywords if keyword in title)
        return count

    def clean_pdf_text(self, text):
        lines = text.splitlines()
        filtered_lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line) and len(line.strip()) > 10]
        text = ' '.join(filtered_lines)
        unwanted_sections = [
            r'(references|bibliography|works cited|author\'s note|author contribution|acknowledgements|disclaimer|declarations|publisher\'s note)', 
            r'(acknowledgment|funding|disclosure|supplementary information|competing interests)'
        ]
        for section in unwanted_sections:
            text = re.split(section, text, flags=re.IGNORECASE)[0]
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_pdf_file(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = ''.join([page.extract_text() for page in reader.pages])
            return self.clean_pdf_text(text)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def clean_text_json(self, text, doi):
        text = text.lower()
        if 'image-web-pdf' in text:
            text = re.split(r'image-web-pdf', text, flags=re.IGNORECASE)[-1]
        doi_fragment = doi.split('doi.org/')[-1]
        if doi_fragment in text[:1000]:
            text = re.split(doi_fragment, text, flags=re.IGNORECASE)[-1]
        patterns_to_remove = [
            r'https?://\S+', r'\b\S+\.(pdf|png|jpg|gif|jpeg|sml|lrg)\b', r'\bimage-(web-pdf|thumbnail|downsampled|high-res)\b',
            r'\baltimg\b', r'\d{1,3}x\d{1,4}', r'\bfig\.\s*\d+\b', r'(true\s+)?sml\s+\d+\s+\d+\s+\d+', r'(true\s+)?jpg\s+\d+\s+\d+\s+\d+',
            r'fx\d+\s+gr\d+', r'gr\d+', r'(?:gif\s+\d+\s+\d+\s+\d+\s+si\d+\s*)+', r'pp\.\s*\d{1,3}(?:-\d{1,3})?\.\s*\d{4}'
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)
        text = re.split(r'(references|bibliography|works cited)', text, flags=re.IGNORECASE)[0]
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_json_file(self, file_path, doi):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        text = data.get('originalText', '')
        return self.clean_text_json(text, doi)
    
    def calculate_keyword_density(self, text):
        words = text.split()
        word_count = len(words)
        if word_count < 100:
            return 0
        keyword_count = sum(text.count(keyword) for keyword in self.keywords)
        return keyword_count / word_count
    
    def save_text(self, file_path, text):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
    
    def load_papers(self):
        papers_to_load = self.filter_papers()
        for _, row in papers_to_load.iterrows():
            paper_id = row['id']
            doi = row['doi']
            json_file_path = os.path.join(self.data_folder, f"{paper_id}.json")
            pdf_file_path = os.path.join(self.data_folder, f"{paper_id}.pdf")
            
            if os.path.exists(json_file_path):
                try:
                    cleaned_text = self.load_json_file(json_file_path, doi)
                    self.papers[paper_id] = cleaned_text
                except (EOFError, json.JSONDecodeError) as e:
                    print(f"Error reading JSON file for ID {paper_id}: {e}")
                    continue
            elif os.path.exists(pdf_file_path):
                content = self.load_pdf_file(pdf_file_path)
                if content:
                    self.papers[paper_id] = content
                else:
                    print(f"Could not load PDF content for ID {paper_id}.")
                    continue
            else:
                print(f"File for ID {paper_id} not found.")
                continue

            density = self.calculate_keyword_density(self.papers[paper_id])
            self.keyword_densities[paper_id] = density

        self.save_papers_by_density()

    def save_papers_by_density(self):
        densities = list(self.keyword_densities.values())
        mean_density = np.mean(densities)
        
        for paper_id, content in self.papers.items():
            density = self.keyword_densities[paper_id]
            word_count = len(content.split())
            if word_count > 250:
                if density > mean_density:
                    self.save_text(os.path.join(self.high_relevant_dir, f"{paper_id}.txt"), content)
                elif 0 < density <= mean_density:
                    self.save_text(os.path.join(self.relevant_dir, f"{paper_id}.txt"), content)
            else:
                print(f"Skipping file {paper_id} due to insufficient word count: {word_count} words.")

# Usage Example:
metadata_file = 'filtered_metadata.csv'
keywords = ["thermoplastic", "automotive", "automobile", "density", "elongation at break", "extrusion", 
            "flexural modulus", "flexural strength", "mechanical properties", "melt mass flow rate", 
            "tensile strength at break", "virgin plastic", "compound", "compare", "valve", "parts", 
            "elongation at yield", "injection moulding", "materials testing", "young's modulus", "stress", 
            "strain", "post-consumer", "post-industrial", "processing", "recyclate", "recyclates", 
            "temperature", "tensile strength", "tensile strength at yield", "comparison", "flow rate", 
            "vicat softening temperature", "MVR", "Melt volume flow rate", "poly"]

data_folder = '/path/to/data'
output_folder = '/path/to/output'

processor = PaperProcessor(metadata_file, keywords, data_folder, output_folder)
processor.load_papers()
