import pandas as pd
from myth_detector import add_myth_detection_features

def main():
    print(" Adding Myth Detection to NASA Climate Dataset")
    print("=" * 50)
    
    # Load your existing dataset
    try:
        df = pd.read_csv('nasa_climate_with_topics.csv')
        print(f" Loaded dataset: {len(df)} comments")
        print(f" Current columns: {len(df.columns)}")
    except FileNotFoundError:
        print(" File 'nasa_climate_with_topics.csv' not found!")
        print("Make sure the file is in the same folder as this script.")
        return
    
    # Show current columns
    print(f"\n Current columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Check if myth detection already exists
    if 'myths_detected' in df.columns:
        print(" Myth detection columns already exist!")
        user_input = input("Reprocess anyway? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    # Add myth detection features
    print(f"\n Processing {len(df)} comments for climate myths...")
    print("This may take a few seconds...")
    
    try:
        df_with_myths = add_myth_detection_features(df)
        
        # Save the enhanced dataset
        output_file = 'nasa_climate_with_myths.csv'
        df_with_myths.to_csv(output_file, index=False)
        
        print(f"\nEnhanced dataset saved as: {output_file}")
        print(f"Dataset now has {len(df_with_myths.columns)} columns (was {len(df.columns)})")
        
        # Show NEW columns added
        new_columns = [col for col in df_with_myths.columns if col not in df.columns]
        print(f"\n New columns added:")
        for col in new_columns:
            print(f"   • {col}")
        
        # Show results summary
        total_myths = df_with_myths['myths_detected'].sum()
        comments_with_myths = df_with_myths['has_myths'].sum()
        high_priority = df_with_myths['high_priority_myth'].sum()
        
        print(f"\n MYTH DETECTION RESULTS:")
        print(f"   • Total myths detected: {total_myths}")
        print(f"   • Comments with myths: {comments_with_myths} ({comments_with_myths/len(df)*100:.1f}%)")
        print(f"   • High priority responses needed: {high_priority}")
        
        # Show sample detected myths
        if comments_with_myths > 0:
            print(f"\n SAMPLE DETECTED MYTHS:")
            myth_samples = df_with_myths[df_with_myths['has_myths']].head(3)
            
            for i, (idx, row) in enumerate(myth_samples.iterrows(), 1):
                print(f"\n   {i}. Comment: {row['text'][:80]}...")
                print(f"      Myths detected: {row['myths_detected']}")
                print(f"      Severity: {row['myth_severity']}")
        else:
            print(f"\n No climate myths detected in this dataset!")
        
        print(f"\n Next step: Update your dashboard to use '{output_file}'")
        
    except Exception as e:
        print(f" Error during processing: {e}")
        print("Check that myth_detector.py is working correctly.")

if __name__ == "__main__":
    main()