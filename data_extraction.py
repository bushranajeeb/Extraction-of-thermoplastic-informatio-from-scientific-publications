import argparse
import pandas as pd
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from material_property_finder import MaterialPropertyFinder
from token_processor import TokenProcessor
from core_components import SentenceProcessor, AutomotivePartFinder
from material_processor import MaterialProcessor
from preprocess import PaperProcessor

class DataExtraction:
    def __init__(self, model_name, fibre_file, abbreviation_dict=None):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        self.fibre_file = 'fibres.csv'  # Path to fibres file
        self.fibre_df = pd.read_csv(self.fibre_file, header=None, dtype=str)

        self.fibre_df = pd.read_csv(fibre_file, header=None, dtype=str)
        self.fibre_dict = {}
        for _, row in self.fibre_df.iterrows():
            full_name = row[0]
            abbreviation = row[1]
            if isinstance(full_name, str) and full_name.strip():
                self.fibre_dict[full_name.strip().lower()] = 'Reinforcement material'
            if isinstance(abbreviation, str) and abbreviation.strip():
                self.fibre_dict[abbreviation.strip().lower()] = 'Reinforcement material'

        self.material_abbreviation_file = 'material_abbreviations.csv'

        # Initialize processors and finders
        self.token_processor = TokenProcessor(fibre_dict=self.fibre_dict, abbreviation_dict=abbreviation_dict)
        self.sentence_processor = SentenceProcessor()
        self.property_finder = MaterialPropertyFinder()
        self.material_processor = MaterialProcessor(fibre_dict=self.fibre_dict, sentence_processor=self.sentence_processor, abbreviation_file=self.material_abbreviation_file)
        self.automotive_part_finder = AutomotivePartFinder()

        # Store last detected material
        self.last_material = None

    def extract_entities(self, text):
        ner_results = self.ner_pipeline(text)
        label_map = {
            'LABEL_0': 'O',
            'LABEL_1': 'MATERIAL',
            'LABEL_2': 'MATERIAL_PROPERTY',
            'LABEL_3': 'PROP_VALUE',
            'LABEL_4': 'AUTOMOTIVE_PART',
            'LABEL_5': 'POLYMER_FAMILY',
            'LABEL_6': 'ORGANIC',
            'LABEL_7': 'INORGANIC',
            'LABEL_8': 'MONOMER',
            'LABEL_9': 'MATERIAL_AMOUNT'
        }
        tokens = []
        for entity in ner_results:
            tokens.append({
                'word': entity['word'],
                'label': label_map.get(entity['entity'], entity['entity']),
                'confidence': entity['score']
            })
        return tokens
    
    def merge_materials(self,structured_data):
        """
        Merge materials with the same name, combining their properties, values, and automotive parts.
        Also ensure that the `Material_compared` field is handled correctly.
        """
        material_dict = {}

        for item in structured_data:
            material = item['Material']
            material_name = material['Name'].lower()

            # If the material is already in the dictionary, update its properties and values
            if material_name in material_dict:
                existing_material = material_dict[material_name]

                # Merge properties
                existing_material['Properties'] = list(set(existing_material['Properties'] + material.get('Properties', [])))

                # Merge property values
                existing_material['PropertyValues'] = list(set(existing_material['PropertyValues'] + material.get('PropertyValues', [])))

                # Merge automotive parts
                if material.get('AutomotivePart'):
                    existing_material['AutomotivePart'] = material['AutomotivePart']

                # Merge compared material (only if it exists and is not a duplicate)
                if material.get('Material_compared') and material['Material_compared'] != existing_material.get('Material_compared'):
                    existing_material['Material_compared'] = material['Material_compared']
        
            # If material is not in the dictionary, add it
            else:
                material_dict[material_name] = {
                    'Name': material['Name'],
                    'ReinforcingMaterial': material['ReinforcingMaterial'],
                    'Amount': material['Amount'],
                    'Properties': material.get('Properties', []),
                    'PropertyTrait': material.get('PropertyTrait'),
                    'PropertyValues': material.get('PropertyValues', []),
                    'AutomotivePart': material.get('AutomotivePart'),
                    'Material_compared': material.get('Material_compared')  # Save the Material_compared field
                }

        # Convert dictionary back to a list format
        return [{'Material': value} for value in material_dict.values()]


    def process_materials(self, text, output_json_file="structured_data.json"):
        # Split text into sentences first
        sentences = self.sentence_processor.process_sentence(text)

        structured_data = []
        self.last_material = None  # Track the last detected material

        for sentence in sentences:
            print(f"Processing sentence: {sentence}")

            # Extract tokens for the current sentence
            tokens = self.extract_entities(sentence)
            tokens = self.token_processor.process(tokens)

            # Process materials for each tokenized sentence
            processed_materials = self.material_processor.process(tokens)

            if processed_materials:
                # If a material is found, update the last_material
                self.last_material = processed_materials[0]  # Assume first material is the main one
                structured_data.extend(processed_materials)  # Use extend instead of append to avoid nested lists

            # Check for property-value pairs and assign the last material if found
            closest_properties, closest_trait, closest_values = self.property_finder.find_closest_property_in_sentence(tokens)
            automotive_part = self.automotive_part_finder.find_automotive_part(tokens)

            if not processed_materials and (closest_properties or closest_values or automotive_part):
                # If no material is found, use the last detected material
                if self.last_material:
                    structured_data.append({
                        'Material': {
                            'Name': self.last_material['Material']['Name'],
                            'ReinforcingMaterial': self.last_material['Material']['ReinforcingMaterial'],
                            'Amount': self.last_material['Material']['Amount'],
                            'Properties': closest_properties,
                            'PropertyTrait': closest_trait,
                            'PropertyValues': closest_values,
                            'AutomotivePart': automotive_part
                        }
                    })

        # Merge the materials to avoid duplication
        merged_materials = self.merge_materials(structured_data)

        # Save the merged structured data to a JSON file
        with open(output_json_file, 'w') as f:
            json.dump(merged_materials, f, indent=4)

        return sentences, merged_materials

def main():
    parser = argparse.ArgumentParser(description='Process some PDFs and extract data using a model.')
    parser.add_argument('--model_name', type=str, required=True, help='The name or path of the fine-tuned model')
    parser.add_argument('--pdf_file', type=str, required=True, help='Path to the PDF file to process')

    args = parser.parse_args()

    # Initialize the NER and Material Processing pipeline
    pipeline = DataExtraction(model_name=args.model_name)

    # Load the text from the PDF
    text = PaperProcessor.load_pdf_file(args.pdf_file)

    # Extract and process material entities from the text
    sentences, structured_data = pipeline.process_materials(text)

    # Output the structured data (optional: you can save it to a file as well)
    print("Extracted Sentences: ", sentences)
    print("Structured Data: ", structured_data)

if __name__ == '__main__':
    main()
