class MaterialPropertyFinder:
    def __init__(self, proximity_range=5, property_traits=None, min_confidence=0.8):
        self.proximity_range = proximity_range
        if property_traits is None:
            self.property_traits = ['high', 'low', 'good', 'poor', 'strong', 'weak', 'excellent', 'increased', 'decreased']
        else:
            self.property_traits = property_traits
        self.min_confidence = min_confidence  # Set confidence threshold (default 80%)

    def find_closest_property_in_sentence(self, sentence_tokens, material_item=None):
        """
        Find the closest MATERIAL_PROPERTY and corresponding PROP_VALUE in the same sentence.
        Handles cases with 'respectively' where multiple properties and values are listed.
        If no material is provided, capture the property-value pair independently.
        Only consider MATERIAL_PROPERTY and PROP_VALUE tokens with confidence higher than the set threshold.
        """
        closest_properties = []
        closest_values = []
        closest_trait = None

        # Only consider MATERIAL_PROPERTY tokens with confidence >= min_confidence
        property_indexes = [i for i, token in enumerate(sentence_tokens) 
                            if token['label'] == 'MATERIAL_PROPERTY' and token['confidence'] >= self.min_confidence]

        # Only consider PROP_VALUE tokens with confidence >= min_confidence
        value_indexes = [i for i, token in enumerate(sentence_tokens) 
                         if token['label'] == 'PROP_VALUE' and token['confidence'] >= self.min_confidence]

        # Handle "respectively" case
        if any("respectively" in token['word'].lower() for token in sentence_tokens):
            closest_properties = [sentence_tokens[i]['word'] for i in property_indexes]
            closest_values = [sentence_tokens[i]['word'] for i in value_indexes]
        else:
            # Normal case: find the closest property and value pairs
            for prop_idx in property_indexes:
                closest_properties.append(sentence_tokens[prop_idx]['word'])

                # Find the corresponding value after the property
                for j in range(prop_idx + 1, min(len(sentence_tokens), prop_idx + self.proximity_range)):
                    if sentence_tokens[j]['label'] == 'PROP_VALUE' and sentence_tokens[j]['confidence'] >= self.min_confidence:
                        closest_values.append(sentence_tokens[j]['word'])
                        break

        if not material_item:
            return closest_properties, closest_trait, closest_values

        return closest_properties, closest_trait, closest_values
