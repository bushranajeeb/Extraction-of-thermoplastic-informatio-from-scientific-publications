import re
import Levenshtein
import pandas as pd

class TokenProcessor:
    def __init__(self, fibre_dict, abbreviation_dict):
        self.fibre_dict = fibre_dict

    def combine_subtokens(self, tokens):
        """
        Combine adjacent tokens that have the same label (except 'O') and handle subtokens with '##'.
        Combine subtokens with '##' for any label, including 'O', but do not combine normal 'O' tokens.
        Example: '30' + '%' -> '30%', 'Glass' + 'Fiber' -> 'Glass Fiber', 'poly' + '##eth' -> 'polyeth'.
        """
        combined_tokens = []
        current_token = ""
        current_label = None
        current_confidence_sum = 0
        token_count = 0
        
        for token in tokens:
            label = token['label']
            word = token['word']
            confidence = token['confidence']
            
            if label == current_label:  # Combine if the current label matches
                if word.startswith('##'):  # Combine subtokens for any label
                    current_token += word[2:]  # Remove the '##' and combine
                    current_confidence_sum += confidence
                    token_count += 1
                elif label != 'O':  # Combine normal tokens for non-'O' labels
                    current_token += " " + word
                    current_confidence_sum += confidence
                    token_count += 1
                else:
                    # If it's 'O' and not a subtoken, add the previous token and start a new one
                    combined_tokens.append({
                        'word': current_token.strip(),
                        'label': current_label,
                        'confidence': current_confidence_sum / token_count
                    })
                    current_token = word  # Start new token for 'O'
                    current_confidence_sum = confidence
                    token_count = 1
            else:
                # Append the previous token if it exists
                if current_token:
                    combined_tokens.append({
                        'word': current_token.strip(),
                        'label': current_label,
                        'confidence': current_confidence_sum / token_count
                    })
                # Start a new token group for the new label
                current_token = word if not word.startswith('##') else word[2:]
                current_label = label
                current_confidence_sum = confidence
                token_count = 1

        # Append the last token if it exists
        if current_token:
            combined_tokens.append({
                'word': current_token.strip(),
                'label': current_label,
                'confidence': current_confidence_sum / token_count
            })

        return combined_tokens

    
    def process_delimiters(self, tokens):
        """
        Handle blend detection based on delimiters like /, +, or - between 'MATERIAL' tokens.
        Example: 'PA', '/', 'PP6' -> 'PA/PP6' as a single 'MATERIAL' token.
        """
        delimiters = ['/', '+', '-']  # List of delimiters to check for blends
        processed_tokens = []
        current_blend = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if current token is a delimiter and the previous and next tokens are 'MATERIAL'
            if token['word'] in delimiters:
                if i > 0 and i < len(tokens) - 1:  # Ensure there are tokens before and after the delimiter
                    prev_token = tokens[i - 1]
                    next_token = tokens[i + 1]
                    
                    # Check if both previous and next tokens are labeled 'MATERIAL'
                    if prev_token['label'] == 'MATERIAL' and next_token['label'] == 'MATERIAL':
                        # Combine the previous, current (delimiter), and next tokens
                        blend_token = f"{prev_token['word']}{token['word']}{next_token['word']}"
                        avg_confidence = (prev_token['confidence'] + next_token['confidence']) / 2  # Average confidence
                        processed_tokens.append({'word': blend_token, 'label': 'MATERIAL', 'confidence': avg_confidence})
                        
                        # Skip the next token since it's already combined
                        i += 2
                        continue
            
            # If not part of a blend, handle normal token processing
            if current_blend:
                # End of a potential blend; append combined tokens
                processed_tokens.append({'word': '/'.join(current_blend), 'label': 'MATERIAL'})
                current_blend = []
            
            # Append the current token if it wasn't a part of a blend
            processed_tokens.append(token)
            i += 1
        
        # Append any remaining tokens in current_blend
        if current_blend:
            processed_tokens.append({'word': '/'.join(current_blend), 'label': 'MATERIAL'})
        
        return processed_tokens    

    def process(self, tokens):
        """
        Main processing function to combine tokens, handle blends, abbreviations, roles, and normalize.
        """
        #print(f'Tokens before combining tokens: {tokens}')
        tokens = self.combine_subtokens(tokens)
        #print(f'Tokens before processing delimiters: {tokens}')
        tokens = self.process_delimiters(tokens)

        return tokens