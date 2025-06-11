import pandas as pd

class ClimateMythDetector:
    """Detect and analyze climate myths in comments"""
    
    def __init__(self):
        self.myth_database = {
            'co2_plant_food': {
                'triggers': [
                    'co2 is plant food', 'co2 is good for plants', 'plants need co2',
                    'more co2 is better', 'co2 helps plants grow', 'co2 benefits plants'
                ],
                'myth_type': 'CO2 Misconception',
                'counter_fact': 'While plants use CO2, NASA research shows higher concentrations reduce nutritional value and increase heat stress, offsetting growth benefits.',
                'nasa_source': 'NASA Goddard Institute for Space Studies',
                'severity': 'medium'
            },
            
            'sun_warming': {
                'triggers': [
                    'sun causes warming', 'solar activity', 'sun cycles', 'solar variations',
                    'sunspots cause climate', 'its the sun not co2'
                ],
                'myth_type': 'Solar Attribution',
                'counter_fact': 'NASA Total Solar Irradiance data shows solar output has been declining while Earth warms - opposite correlation expected if sun was the cause.',
                'nasa_source': 'NASA Solar Dynamics Observatory',
                'severity': 'high'
            },
            
            'natural_cycles': {
                'triggers': [
                    'natural cycles', 'climate always changed', 'natural variation',
                    'earth naturally warms', 'natural climate change', 'always been changing'
                ],
                'myth_type': 'Natural Variation',
                'counter_fact': 'Current warming is 10x faster than typical post-ice age recovery. NASA data shows human influence is the dominant factor.',
                'nasa_source': 'NASA Climate Change Evidence',
                'severity': 'medium'
            },
            
            'climate_hoax': {
                'triggers': [
                    'climate hoax', 'climate scam', 'climate lie', 'fake climate',
                    'climate conspiracy', 'scientists lying', 'climate fraud'
                ],
                'myth_type': 'Conspiracy Theory',
                'counter_fact': '97% of actively publishing climate scientists agree on human-caused warming. NASA provides transparent data and methodology.',
                'nasa_source': 'NASA Scientific Consensus',
                'severity': 'high'
            },
            
            'ice_age_coming': {
                'triggers': [
                    'ice age coming', 'cooling not warming', 'getting colder',
                    'mini ice age', 'global cooling', 'earth is cooling'
                ],
                'myth_type': 'Cooling Myth',
                'counter_fact': 'NASA temperature records show clear warming trend. No credible evidence supports imminent cooling.',
                'nasa_source': 'NASA GISS Temperature Analysis',
                'severity': 'medium'
            }
        }
    
    def detect_myths(self, text):
        """Detect climate myths in a text"""
        if not text or pd.isna(text):
            return {
                'myths_detected': [],
                'myth_count': 0,
                'highest_severity': 'none',
                'requires_response': False
            }
        
        text_lower = str(text).lower()
        detected_myths = []
        
        for myth_id, myth_data in self.myth_database.items():
            for trigger in myth_data['triggers']:
                if trigger.lower() in text_lower:
                    detected_myths.append({
                        'myth_id': myth_id,
                        'myth_type': myth_data['myth_type'],
                        'trigger_phrase': trigger,
                        'counter_fact': myth_data['counter_fact'],
                        'nasa_source': myth_data['nasa_source'],
                        'severity': myth_data['severity']
                    })
                    break  # Only count each myth once per comment
        
        # Determine highest severity
        severities = [myth['severity'] for myth in detected_myths]
        severity_order = {'low': 1, 'medium': 2, 'high': 3}
        highest_severity = 'none'
        
        if severities:
            highest_severity = max(severities, key=lambda x: severity_order.get(x, 0))
        
        return {
            'myths_detected': detected_myths,
            'myth_count': len(detected_myths),
            'highest_severity': highest_severity,
            'requires_response': len(detected_myths) > 0
        }
    
    def generate_response(self, detected_myths):
        """Generate educational response to myths"""
        if not detected_myths:
            return None
        
        # Use the first detected myth for response
        primary_myth = detected_myths[0]
        
        response = f"""Thanks for engaging with NASA's climate content! 
        
Regarding {primary_myth['myth_type'].lower()}: {primary_myth['counter_fact']}

For more information, check out NASA's research at: {primary_myth['nasa_source']}

We appreciate your interest in climate science! 🌍"""
        
        return response

# FUNCTION IS NOW OUTSIDE THE CLASS (correct indentation):
def add_myth_detection_features(df):
    """Add climate myth detection to existing dataset"""
    print("\nAdding Climate Myth Detection...")

    myth_detector = ClimateMythDetector()

    # Apply myth detection to all comments
    myth_results = df['text'].apply(myth_detector.detect_myths)

    # Extract results into separate columns
    df = df.copy()  # Don't modify original
    df['myths_detected'] = myth_results.apply(lambda x: x['myth_count'])
    df['myth_severity'] = myth_results.apply(lambda x: x['highest_severity'])
    df['requires_response'] = myth_results.apply(lambda x: x['requires_response'])
    df['myth_details'] = myth_results.apply(lambda x: x['myths_detected'])

    # Create myth categories
    df['has_myths'] = df['myths_detected'] > 0
    df['high_priority_myth'] = df['myth_severity'] == 'high'

    # Summary statistics
    total_myths = df['myths_detected'].sum()
    comments_with_myths = df['has_myths'].sum()
    high_priority = df['high_priority_myth'].sum()

    print(f"✅ Myth Detection Complete:")
    print(f"   • Total myths detected: {total_myths}")
    print(f"   • Comments containing myths: {comments_with_myths} ({comments_with_myths/len(df)*100:.1f}%)")
    print(f"   • High priority responses needed: {high_priority}")

    return df

# Test code
if __name__ == "__main__":
    detector = ClimateMythDetector()
    test_result = detector.detect_myths("co2 is plant food")
    print("ClimateMythDetector test successful!")
    print("Test result:", test_result)