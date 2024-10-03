import pandas as pd
from base_classes import AutomotivePartFinder
from material_property_finder import MaterialPropertyFinder
import Levenshtein

class MaterialProcessor:
    def __init__(self, fibre_dict, sentence_processor, abbreviation_file):
        self.fibre_dict = fibre_dict
        self.sentence_processor = sentence_processor
        self.material_abbreviation_dict = self.load_material_abbreviations(abbreviation_file)
        self.material_property_finder = MaterialPropertyFinder()  # Initialize the property finder
        self.automotive_part_finder = AutomotivePartFinder()  # Initialize the automotive part finder
        self.context = {}  # Store context for materials across sentences
        self.comparison_traits = [
        'higher than', 'lower than', 'better than', 'worse than', 'more than', 'less than',
        'compared to', 'in contrast to', 'higher', 'lower', 'better', 'whereas', 'while',
        'as opposed to', 'both', 'on the other hand', 'similar to', 'different from',
        'contrary to', 'increased', 'decreased', 'compared', 'more', 'less'
    ]

    def load_material_abbreviations(self, file_path):
        """
        Load the material abbreviations file into a dictionary.
        The dictionary will map abbreviations and synonyms to the full material name.
        """
        df = pd.read_csv(file_path)
        material_dict = {}
        
        # Iterate through the CSV file and populate the dictionary
        for _, row in df.iterrows():
            full_name = row['FullForm'].strip().lower()
            
            # Add abbreviation and synonyms mapped to the full name
            material_dict[row['Abbreviation'].strip().lower()] = full_name
            if pd.notna(row['Synonyms']):
                synonyms = [syn.strip().lower() for syn in row['Synonyms'].split(',')]
                for synonym in synonyms:
                    material_dict[synonym] = full_name
        
        return material_dict

    def detect_reinforcement_materials(self, tokens):
        """
        Detect if a material is reinforced with a fiber and capture the MATERIAL_AMOUNT.
        This function works within each sentence.
        """
        reinforcement_keywords = ['reinforced', 'reinforcement']
        abbreviation_map = {'gf': 'glass fibre'}  # Add common abbreviations for fibers
        detected_materials = []
        proximity_range = 5  # Range to capture tokens around the reinforcement word
        current_material = None
        reinforcing_material = None
        material_amount = None
        distance_threshold = 2  # Threshold for Levenshtein distance to consider a close match

        # List of stopwords to avoid false positives
        stopwords = set(['with', 'has', 'is', 'than', 'the', 'a', 'an', 'and'])

        for i, token in enumerate(tokens):
            word = token['word'].lower().strip()

            # Check if the token is a MATERIAL and exclude "thermoplastic" if it appears alone
            if token['label'] == 'MATERIAL' and word != 'thermoplastic':
                current_material = token['word']

            # Check if the token is a MATERIAL_AMOUNT (e.g., "30%")
            if token['label'] == 'MATERIAL_AMOUNT':
                material_amount = token['word']

            # Check for reinforcement keywords
            if word in reinforcement_keywords:
                # Check proximity for fiber material
                for j in range(max(0, i - proximity_range), min(len(tokens), i + proximity_range)):
                    nearby_token = tokens[j]
                    nearby_word = nearby_token['word'].lower().strip()

                    # Check for common abbreviations (e.g., "GF" for "Glass Fibre")
                    if nearby_word in abbreviation_map:
                        reinforcing_material = abbreviation_map[nearby_word]
                        break

                    # Skip stopwords and short words that are likely irrelevant
                    if nearby_word in stopwords or len(nearby_word) <= 2:
                        continue

                    # Handle multi-word fibers, e.g., "glass fibre"
                    combined_fibre = nearby_word
                    if j + 1 < len(tokens):
                        next_word = tokens[j + 1]['word'].lower().strip()
                        combined_fibre = f"{nearby_word} {next_word}"

                    # Check if the combined fiber is in the fibre_dict
                    if combined_fibre in self.fibre_dict:
                        reinforcing_material = combined_fibre
                        break

                    # Check Levenshtein distance for close matches on individual words
                    for fibre in self.fibre_dict.keys():
                        if len(nearby_word) >= 3 and Levenshtein.distance(nearby_word, fibre) <= distance_threshold:
                            reinforcing_material = fibre
                            break

                # If we found both a material and reinforcing fiber, store the pair
                if current_material:
                    detected_materials.append({
                        'Material': {
                            'Name': current_material,
                            'ReinforcingMaterial': reinforcing_material,
                            'Amount': material_amount  # Store the amount with the material
                        }
                    })
                    # Reset after pairing
                    current_material = None
                    reinforcing_material = None
                    material_amount = None

        return detected_materials

    
    def combine_material_mentions(self, tokens):
        """
        Detect and combine duplicate material mentions (e.g., POM (Polyoxymethylene)).
        """
        combined_tokens = []
        skip_next = False

        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue

            # Check for material + parentheses pattern (e.g., POM (Polyoxymethylene))
            if token['label'] == 'MATERIAL' and i + 2 < len(tokens):
                if tokens[i + 1]['word'] == '(' and tokens[i + 2]['label'] == 'MATERIAL':
                    # Combine materials into one token (e.g., POM (Polyoxymethylene))
                    combined_material = f"{token['word']} ({tokens[i + 2]['word']})"
                    combined_tokens.append({'word': combined_material, 'label': 'MATERIAL', 'confidence': token['confidence']})
                    skip_next = True  # Skip the next two tokens (parentheses and synonym)
                    continue

            # Otherwise, append the token as is
            combined_tokens.append(token)

        return combined_tokens

    def normalize_material(self, token):
        """
        Normalize a material token by checking if it's an abbreviation or synonym and replacing it with the full name.
        """
        word = token['word'].strip().lower()
        normalized_word = self.material_abbreviation_dict.get(word, word)

        # Combine "POM (Polyoxymethylene)" into "Polyoxymethylene"
        if 'pom' in normalized_word:
            return 'polyoxymethylene'  # Normalize to the full material name
        return normalized_word
    
    def detect_comparison(self, sentence_tokens):
        """
        Detects if there is a comparison between two materials in the sentence.
        If found, it returns the second material as `Material_compared`.
        """
        materials = [token['word'] for token in sentence_tokens if token['label'] == 'MATERIAL']
        comparison_found = any(trait in ' '.join([token['word'] for token in sentence_tokens]).lower() for trait in self.comparison_traits)

        # If we found at least two materials and a comparison trait, link the second material as `Material_compared`
        if len(materials) >= 2 and comparison_found:
            return materials[1]  # The second material is the compared one

        return None

    def process(self, tokens):
        """
        Process tokens to detect materials, their reinforcement, amount, properties, and comparisons.
        This method processes tokens for a single sentence and stores context to associate materials with properties across the document.
        """
        # Final list of materials
        final_materials = []
        material_set = set()

        # First, normalize and combine materials within the tokens
        tokens = self.combine_material_mentions(tokens)
        for token in tokens:
            if token['label'] == 'MATERIAL':
                token['word'] = self.normalize_material(token)

        # Detect reinforcement materials and amounts within the current sentence
        processed_materials = self.detect_reinforcement_materials(tokens)

        # If no material with reinforcement is found, process all materials
        if not processed_materials:
            # Detect all materials without reinforcement
            processed_materials = [{'Material': {
                'Name': token['word'],
                'ReinforcingMaterial': None,
                'Amount': None
            }} for token in tokens if token['label'] == 'MATERIAL']

        # Find the closest properties for each detected material
        for material in processed_materials:
            material_name = material['Material']['Name'].lower()
            # Avoid duplicates in the final materials list
            if material_name not in material_set:
                # Find closest material property and its trait/value
                closest_properties, closest_trait, closest_values = self.material_property_finder.find_closest_property_in_sentence(tokens, material)

                # Add the property and trait/value to the material object
                material['Material']['Properties'] = closest_properties
                material['Material']['PropertyTrait'] = closest_trait
                material['Material']['PropertyValues'] = closest_values

                # Detect automotive part in the tokens
                automotive_part = self.automotive_part_finder.find_automotive_part(tokens)
                if automotive_part:
                    material['Material']['AutomotivePart'] = automotive_part

                # Check if the material is being compared to another material
                material_compared = self.detect_comparison(tokens)
                if material_compared:
                    material['Material']['Material_compared'] = material_compared

                final_materials.append(material)
                material_set.add(material_name)

        return final_materials