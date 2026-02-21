import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing code
combined_df = pd.read_csv('combined_embeddings_data.csv')
themes_df = pd.read_csv('outputs/output_text_to_themes_mapping.csv')
combined_df['text_lower'] = combined_df['text'].str.lower()
combined_df = combined_df[['text_lower']].rename(columns = {'text_lower': 'text'})
themes_df = themes_df[['text', 'part1', 'part2']]

## Corrects spelling error where some instances of 'Horny' are written as 'Horn' in the data
themes_df['part1'] = themes_df['part1'].str.replace('Horn', 'Horny').str.replace('Hornyy', 'Horny')

all_themes_df = combined_df.merge(themes_df, on = 'text')
summary_counts = all_themes_df.groupby(['part1']).count()
summary_counts = pd.DataFrame(summary_counts).sort_values('text', ascending = False)
ix = summary_counts['text'] >= 5
summary_counts_filtered = summary_counts.loc[ix]

# Professional bar chart
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8')

# Create the bar chart
bars = plt.bar(range(len(summary_counts_filtered)),
               summary_counts_filtered['text'],
               color='steelblue',
               alpha=0.8,
               edgecolor='black',
               linewidth=0.5)

# Professional formatting
plt.title('Deepseek Parts Frequencies', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Themes', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)

# Set x-axis labels (theme names)
plt.xticks(range(len(summary_counts_filtered)),
           summary_counts_filtered.index,
           rotation=45,
           ha='right')

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--', axis='y')

# Clean layout
plt.tight_layout()

# Show the plot
plt.show()

# Save as high-quality PNG
plt.savefig('outputs/theme_frequency_deepseek_1000_longest.png', dpi=300, bbox_inches='tight')

# Your existing saves
summary_counts.to_csv('outputs/parts_counts.csv')
pd.DataFrame(all_themes_df).to_csv('outputs/all_themes_df.csv')
