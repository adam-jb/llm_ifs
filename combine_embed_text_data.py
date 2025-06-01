import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_combine_data():
    """
    Load all three files and combine them into a single DataFrame for analysis
    """
    print("Loading data files...")

    # Load conversation embeddings
    convo_df = pd.read_csv('embeddings/convo_2d.csv')
    print(f"Conversation embeddings shape: {convo_df.shape}")

    # Load output embeddings
    output_df = pd.read_csv('embeddings/output_2d.csv')
    print(f"Output embeddings shape: {output_df.shape}")

    # Load text data
    with open('outputs/lmsys_deepseek_worker_longest_chats_1000.txt', 'r', encoding='utf-8') as f:
        text_lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Text entries loaded: {len(text_lines)}")

    # Create combined DataFrame
    max_rows = min(len(convo_df), len(output_df), len(text_lines))
    print(f"Using {max_rows} rows for analysis (minimum across all files)")

    combined_df = pd.DataFrame({
        'index': range(max_rows),
        'convo_x': convo_df.iloc[:max_rows, 0],  # Assuming first column is x
        'convo_y': convo_df.iloc[:max_rows, 1],  # Assuming second column is y
        'output_x': output_df.iloc[:max_rows, 0],
        'output_y': output_df.iloc[:max_rows, 1],
        'text': text_lines[:max_rows]
    })

    return combined_df

def analyze_coordinate_differences(df):
    """
    Analyze the differences between conversation and output coordinates
    """
    print("\n" + "="*60)
    print("COORDINATE ANALYSIS")
    print("="*60)

    # Calculate distances between conversation and output points
    df['distance'] = np.sqrt((df['convo_x'] - df['output_x'])**2 +
                            (df['convo_y'] - df['output_y'])**2)

    # Calculate coordinate differences
    df['x_diff'] = df['output_x'] - df['convo_x']
    df['y_diff'] = df['output_y'] - df['convo_y']

    print(f"Distance statistics:")
    print(f"  Mean distance: {df['distance'].mean():.4f}")
    print(f"  Median distance: {df['distance'].median():.4f}")
    print(f"  Min distance: {df['distance'].min():.4f}")
    print(f"  Max distance: {df['distance'].max():.4f}")
    print(f"  Std distance: {df['distance'].std():.4f}")

    print(f"\nX-coordinate differences:")
    print(f"  Mean: {df['x_diff'].mean():.4f}")
    print(f"  Std: {df['x_diff'].std():.4f}")
    print(f"  Range: [{df['x_diff'].min():.4f}, {df['x_diff'].max():.4f}]")

    print(f"\nY-coordinate differences:")
    print(f"  Mean: {df['y_diff'].mean():.4f}")
    print(f"  Std: {df['y_diff'].std():.4f}")
    print(f"  Range: [{df['y_diff'].min():.4f}, {df['y_diff'].max():.4f}]")

    # Check for identical coordinates (potential bug indicator)
    identical_coords = (df['x_diff'] == 0) & (df['y_diff'] == 0)
    print(f"\nPoints with identical coordinates: {identical_coords.sum()} ({identical_coords.mean()*100:.1f}%)")

    return df

def check_text_alignment(df):
    """
    Check for potential text alignment issues
    """
    print("\n" + "="*60)
    print("TEXT ALIGNMENT ANALYSIS")
    print("="*60)

    # Check for duplicate texts
    duplicate_texts = df['text'].duplicated()
    print(f"Duplicate text entries: {duplicate_texts.sum()} ({duplicate_texts.mean()*100:.1f}%)")

    if duplicate_texts.sum() > 0:
        print("\nFirst few duplicate texts:")
        duplicated_df = df[df['text'].duplicated(keep=False)].sort_values('text')
        for i, (idx, row) in enumerate(duplicated_df.head(10).iterrows()):
            print(f"  Index {row['index']}: {row['text'][:100]}...")
            if i >= 4:  # Show max 5 examples
                break

    # Check text length distribution
    df['text_length'] = df['text'].str.len()
    print(f"\nText length statistics:")
    print(f"  Mean length: {df['text_length'].mean():.1f} characters")
    print(f"  Median length: {df['text_length'].median():.1f} characters")
    print(f"  Min length: {df['text_length'].min()} characters")
    print(f"  Max length: {df['text_length'].max()} characters")

    return df

def find_suspicious_patterns(df):
    """
    Look for patterns that might indicate data alignment issues
    """
    print("\n" + "="*60)
    print("SUSPICIOUS PATTERN DETECTION")
    print("="*60)

    # Look for cases where coordinates are very similar but texts are different
    # This could indicate misalignment

    # Group by similar coordinates (rounded to 3 decimal places)
    df['convo_x_round'] = df['convo_x'].round(3)
    df['convo_y_round'] = df['convo_y'].round(3)
    df['output_x_round'] = df['output_x'].round(3)
    df['output_y_round'] = df['output_y'].round(3)

    # Check for multiple different texts at same conversation coordinates
    convo_coord_groups = df.groupby(['convo_x_round', 'convo_y_round'])
    suspicious_convo = []

    for (x, y), group in convo_coord_groups:
        if len(group) > 1:
            unique_texts = group['text'].nunique()
            if unique_texts > 1:
                suspicious_convo.append({
                    'coords': (x, y),
                    'count': len(group),
                    'unique_texts': unique_texts,
                    'indices': group['index'].tolist()
                })

    print(f"Conversation coordinates with multiple different texts: {len(suspicious_convo)}")
    if suspicious_convo:
        print("Examples:")
        for item in suspicious_convo[:3]:
            print(f"  Coords ({item['coords'][0]}, {item['coords'][1]}): {item['count']} entries, {item['unique_texts']} unique texts")
            print(f"    Indices: {item['indices']}")

    # Check for multiple different texts at same output coordinates
    output_coord_groups = df.groupby(['output_x_round', 'output_y_round'])
    suspicious_output = []

    for (x, y), group in output_coord_groups:
        if len(group) > 1:
            unique_texts = group['text'].nunique()
            if unique_texts > 1:
                suspicious_output.append({
                    'coords': (x, y),
                    'count': len(group),
                    'unique_texts': unique_texts,
                    'indices': group['index'].tolist()
                })

    print(f"Output coordinates with multiple different texts: {len(suspicious_output)}")
    if suspicious_output:
        print("Examples:")
        for item in suspicious_output[:3]:
            print(f"  Coords ({item['coords'][0]}, {item['coords'][1]}): {item['count']} entries, {item['unique_texts']} unique texts")
            print(f"    Indices: {item['indices']}")

    return df

def create_diagnostic_plots(df):
    """
    Create plots to visualize the data and potential issues
    """
    print("\n" + "="*60)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Scatter plot of conversation vs output coordinates
    axes[0, 0].scatter(df['convo_x'], df['convo_y'], alpha=0.6, c='red', s=20, label='Conversation')
    axes[0, 0].scatter(df['output_x'], df['output_y'], alpha=0.6, c='blue', s=20, label='Output')
    axes[0, 0].set_xlabel('X Coordinate')
    axes[0, 0].set_ylabel('Y Coordinate')
    axes[0, 0].set_title('Conversation vs Output Coordinates')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distance distribution
    axes[0, 1].hist(df['distance'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Distance between Conversation and Output')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Distances')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: X and Y differences
    axes[1, 0].scatter(df['x_diff'], df['y_diff'], alpha=0.6, s=20)
    axes[1, 0].set_xlabel('X Difference (Output - Conversation)')
    axes[1, 0].set_ylabel('Y Difference (Output - Conversation)')
    axes[1, 0].set_title('Coordinate Differences')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # Plot 4: Text length vs distance
    axes[1, 1].scatter(df['text_length'], df['distance'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Text Length (characters)')
    axes[1, 1].set_ylabel('Distance between Coordinates')
    axes[1, 1].set_title('Text Length vs Coordinate Distance')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Diagnostic plots saved as 'data_validation_plots.png'")

def save_combined_data(df):
    """
    Save the combined data to CSV for further analysis
    """
    # Clean up temporary columns
    columns_to_drop = [col for col in df.columns if col.endswith('_round')]
    df_clean = df.drop(columns=columns_to_drop)

    # Save to CSV
    df_clean.to_csv('combined_embeddings_data.csv', index=False)
    print(f"\nCombined data saved to 'combined_embeddings_data.csv'")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"Shape: {df_clean.shape}")

    return df_clean

def main():
    """
    Main analysis function
    """
    print("EMBEDDING DATA VALIDATION SCRIPT")
    print("="*60)

    try:
        # Load and combine data
        df = load_and_combine_data()

        # Run analyses
        df = analyze_coordinate_differences(df)
        df = check_text_alignment(df)
        df = find_suspicious_patterns(df)

        # Create plots
        create_diagnostic_plots(df)

        # Save results
        df_final = save_combined_data(df)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ Data loaded and combined successfully")
        print("✓ Coordinate analysis completed")
        print("✓ Text alignment checked")
        print("✓ Suspicious patterns detected")
        print("✓ Diagnostic plots created")
        print("✓ Combined data saved")

        print(f"\nNext steps:")
        print(f"1. Review the diagnostic plots")
        print(f"2. Check 'combined_embeddings_data.csv' for detailed data")
        print(f"3. Investigate any suspicious patterns found")

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please ensure all files are in the correct locations:")
        print("  - embeddings/convo_2d.csv")
        print("  - embeddings/output_2d.csv")
        print("  - outputs/lmsys_deepseek_worker_longest_chats_1000.txt")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
