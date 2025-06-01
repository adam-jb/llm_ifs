import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

# Load the combined dataframe (recreate it as in your original code)
print("Loading data...")
df_csv = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
with open('outputs/lmsys_deepseek_worker_longest_chats.txt', 'r', encoding='utf-8') as f:
    text_lines = f.readlines()

# Create text DataFrame
df_txt = pd.DataFrame({
    'output_text': [line.strip() for line in text_lines]
})
df_txt['number'] = range(len(df_txt))
df_txt = df_txt.set_index('number')

# Join dataframes
df_csv = df_csv.set_index('number')
df_combined = df_csv.join(df_txt, how='inner')

print(f"Combined DataFrame shape: {df_combined.shape}")

# Load embeddings
print("\nLoading embeddings...")
with open('embeddings/conversation_embeddings.pkl', 'rb') as f:
    conversation_embeddings = pickle.load(f)
with open('embeddings/output_embeddings.pkl', 'rb') as f:
    output_embeddings = pickle.load(f)

print(f"Conversation embeddings shape: {conversation_embeddings.shape}")
print(f"Output embeddings shape: {output_embeddings.shape}")

# Check for identical texts and their embeddings
def find_identical_texts_and_check_embeddings(df, embeddings, text_column, embedding_name):
    print(f"\n=== Checking {embedding_name} ===")

    # Reset index to ensure we're working with sequential indices
    df_reset = df.reset_index(drop=True)

    # Group by text content to find duplicates
    text_groups = defaultdict(list)
    for idx, text in enumerate(df_reset[text_column]):
        text_groups[text].append(idx)

    # Find groups with more than one occurrence
    duplicate_groups = {text: indices for text, indices in text_groups.items() if len(indices) > 1}

    print(f"Found {len(duplicate_groups)} unique texts with duplicates")
    print(f"Total duplicate instances: {sum(len(indices) for indices in duplicate_groups.values())}")

    # Let's also show some examples of what we found
    print(f"Total unique texts: {len(text_groups)}")
    print(f"Sample texts (first 5): {list(text_groups.keys())[:5]}")

    if not duplicate_groups:
        print("No duplicate texts found!")
        print("Let's debug this...")

        # Debug: show some actual text values
        sample_texts = df_reset[text_column].head(10).tolist()
        print(f"Sample texts from dataframe: {sample_texts}")

        # Check for near-duplicates (case differences, whitespace)
        text_counts = df_reset[text_column].value_counts()
        print(f"Value counts for top duplicates:")
        print(text_counts.head(10))

        return None, None

    # Check embeddings for each duplicate group
    all_identical = True
    embedding_issues = []

    for text, indices in duplicate_groups.items():
        print(f"\nText preview: '{text[:100]}...' (appears {len(indices)} times)")
        print(f"Indices: {indices}")

        # Get embeddings for these indices
        group_embeddings = embeddings[indices]

        # Check if all embeddings in this group are identical
        first_embedding = group_embeddings[0]
        group_identical = True

        for i, embedding in enumerate(group_embeddings[1:], 1):
            if not np.allclose(first_embedding, embedding, rtol=1e-10, atol=1e-12):
                group_identical = False
                all_identical = False
                max_diff = np.max(np.abs(first_embedding - embedding))
                print(f"  ❌ Index {indices[0]} vs {indices[i]}: Max difference = {max_diff}")
                embedding_issues.append({
                    'text_preview': text[:100],
                    'indices': [indices[0], indices[i]],
                    'max_difference': max_diff
                })

        if group_identical:
            print(f"  ✅ All {len(indices)} embeddings are identical")
        else:
            print(f"  ❌ Embeddings differ for identical text!")

    print(f"\n--- {embedding_name} Summary ---")
    if all_identical:
        print("✅ ALL identical texts have identical embeddings!")
    else:
        print(f"❌ Found {len(embedding_issues)} cases where identical texts have different embeddings")

        # Show most problematic cases
        if embedding_issues:
            print("\nWorst cases (largest differences):")
            embedding_issues.sort(key=lambda x: x['max_difference'], reverse=True)
            for issue in embedding_issues[:5]:
                print(f"  Indices {issue['indices']}: difference = {issue['max_difference']:.2e}")
                print(f"    Text: '{issue['text_preview']}...'")

    return all_identical, embedding_issues

# Check conversation embeddings
result = find_identical_texts_and_check_embeddings(
    df_combined, conversation_embeddings, 'conversation_text', 'Conversation Embeddings'
)

if result is not None:
    conv_ok, conv_issues = result
else:
    print("Skipping conversation embeddings due to no duplicates found")
    conv_ok, conv_issues = True, []

# Check output embeddings
result = find_identical_texts_and_check_embeddings(
    df_combined, output_embeddings, 'output_text', 'Output Embeddings'
)

if result is not None:
    output_ok, output_issues = result
else:
    print("Skipping output embeddings due to no duplicates found")
    output_ok, output_issues = True, []

# Overall summary
print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)

if conv_ok and output_ok:
    print("✅ ALL GOOD: Identical texts have identical embeddings!")
    print("The UMAP coordinate differences must be due to other factors.")
else:
    print("❌ PROBLEM FOUND: Some identical texts have different embeddings!")
    if not conv_ok:
        print(f"  - Conversation embeddings: {len(conv_issues)} problematic cases")
    if not output_ok:
        print(f"  - Output embeddings: {len(output_issues)} problematic cases")
    print("\nThis explains why UMAP gives different coordinates for identical texts.")

# Additional check: Look at the actual embedding values for a few duplicate cases
print("\n" + "="*60)
print("DETAILED EMBEDDING INSPECTION")
print("="*60)

# Let's manually check for duplicates in the raw text file you provided
print("Manual duplicate check from the output text...")

# Reset dataframe for clean indexing
df_reset = df_combined.reset_index(drop=True)
output_texts = df_reset['output_text'].tolist()

# Find duplicates manually
text_counts = {}
for i, text in enumerate(output_texts):
    if text in text_counts:
        text_counts[text].append(i)
    else:
        text_counts[text] = [i]

duplicates = {text: indices for text, indices in text_counts.items() if len(indices) > 1}

print(f"Found {len(duplicates)} duplicate texts")
if duplicates:
    # Show a few examples
    for i, (text, indices) in enumerate(list(duplicates.items())[:5]):
        print(f"\nDuplicate {i+1}: '{text}' appears at indices {indices}")

        # Check embeddings for first duplicate
        if len(indices) >= 2:
            idx1, idx2 = indices[0], indices[1]
            emb1 = output_embeddings[idx1]
            emb2 = output_embeddings[idx2]

            identical = np.array_equal(emb1, emb2)
            close = np.allclose(emb1, emb2, rtol=1e-10, atol=1e-12)
            max_diff = np.max(np.abs(emb1 - emb2))

            print(f"  Embeddings identical? {identical}")
            print(f"  Embeddings close? {close}")
            print(f"  Max difference: {max_diff}")

            if not identical:
                print(f"  🔥 PROBLEM: Identical text has different embeddings!")
else:
    print("Still no duplicates found - there might be an issue with data loading")

print("\nDone!")
