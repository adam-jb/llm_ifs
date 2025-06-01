import pandas as pd
import numpy as np
import os
import openai
import pickle
# pip instal umap-learn   ## not pip install umap!
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import math
from tqdm import tqdm
import time

OPENAI_API_KEY = "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"


# Step 1: Read the CSV file
print("Step 1: Reading CSV file...")
df_csv = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
print(f"CSV shape: {df_csv.shape}")
print(f"CSV columns: {df_csv.columns.tolist()}")

# Step 2: Read the text file and create DataFrame
print("\nStep 2: Reading text file and creating DataFrame...")
with open('outputs/lmsys_deepseek_worker_longest_chats.txt', 'r', encoding='utf-8') as f:
    text_lines = f.readlines()

## Store top lines for later
with open('outputs/lmsys_deepseek_worker_longest_chats_1000.txt', 'w', encoding='utf-8') as f:
    f.writelines(text_lines[:1000])

# Create DataFrame with 'number' as index starting from 0
df_txt = pd.DataFrame({
    'output_text': [line.strip() for line in text_lines]
})
df_txt['number'] = range(len(df_txt))
df_txt = df_txt.set_index('number')
print(f"Text file shape: {df_txt.shape}")

# Step 3: Join the DataFrames
print("\nStep 3: Joining DataFrames on 'number'...")
# Ensure the CSV 'number' column is the index
df_csv = df_csv.set_index('number')
# Join on index
df_combined = df_csv.join(df_txt, how='inner')
print(f"Combined DataFrame shape: {df_combined.shape}")
print(f"Combined DataFrame columns: {df_combined.columns.tolist()}")

# Step 4: Get embeddings using OpenAI
print("\nStep 4: Getting embeddings from OpenAI...")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Check your saved embeddings for the known duplicates
# duplicate_texts = ["Curious playful spark", "curious creative spark", "Creative Flow Surge"]


# # Test with a simple 3-word phrase
# import time

# test_text = "Curious playful spark"
# print(f"Testing text: '{test_text}'")

# embeddings = []
# for i in range(5):
#     response = openai_client.embeddings.create(
#         input=[test_text],
#         model="text-embedding-3-large"
#     )
#     emb = response.data[0].embedding
#     embeddings.append(emb)
#     print(f"Embedding {i+1} first 5 values: {emb[:5]}")
#     time.sleep(0.5)  # Small delay

# # Check if any are identical
# for i in range(len(embeddings)):
#     for j in range(i+1, len(embeddings)):
#         identical = np.array_equal(embeddings[i], embeddings[j])
#         max_diff = np.max(np.abs(np.array(embeddings[i]) - np.array(embeddings[j])))
#         print(f"Embeddings {i+1} vs {j+1}: Identical={identical}, Max diff={max_diff:.10f}")


def chunk_text(text, max_tokens=7500):
    """
    Split text into equal-sized chunks for consistent weighting.
    Using rough estimate of 4 characters per token.
    """
    # Rough token estimation (conservative)
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token

    # Calculate optimal chunk size for equal distribution
    text_length = len(text)
    if text_length <= max_chars:
        return [text]

    # Calculate number of chunks needed
    num_chunks = math.ceil(text_length / max_chars)

    # Calculate actual chunk size for equal distribution
    chunk_size = math.ceil(text_length / num_chunks)

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, text_length)
        chunks.append(text[start:end])

    return chunks

cache_file = 'embeddings/embedding_cache.pkl'
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        embedding_cache = pickle.load(f)
    print(f"Loaded cache with {len(embedding_cache)} entries")
else:
    embedding_cache = {}

def get_cached_embedding(text, model="text-embedding-3-large"):
    if text not in embedding_cache:
        response = openai_client.embeddings.create(input=[text], model=model)
        embedding_cache[text] = response.data[0].embedding

        # Save cache periodically
        if len(embedding_cache) % 100 == 0:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding_cache, f)

    return embedding_cache[text]

def get_embeddings_cached(texts, model="text-embedding-3-large"):
    embeddings = []
    for text in tqdm(texts, desc="Getting cached embeddings"):
        emb = get_cached_embedding(text, model)
        embeddings.append(emb)

    # Save final cache
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding_cache, f)

    return np.array(embeddings)


def get_embeddings(texts, model="text-embedding-3-large"):
    """Get embeddings for a list of texts using OpenAI API with equal-weight chunking"""
    import math
    embeddings = []

    # Process each text individually to handle chunking
    for text in tqdm(texts, desc="Processing texts"):

        # For rate limits
        time.sleep(0.1)

        # Check if text needs chunking (conservative estimate)
        if len(text) > 30000:  # ~7500 tokens
            # Split into equal-sized chunks
            chunks = chunk_text(text, max_tokens=7500)

            # Verify no chunk exceeds limits (with safety margin)
            safe_chunks = []
            for chunk in chunks:
                if len(chunk) > 28000:  # If still too long, split further
                    sub_chunks = chunk_text(chunk, max_tokens=6000)
                    safe_chunks.extend(sub_chunks)
                else:
                    safe_chunks.append(chunk)

            # Get embeddings for each chunk
            chunk_embeddings = []
            for chunk in safe_chunks:
                try:
                    response = openai_client.embeddings.create(
                        input=[chunk],
                        model=model
                    )
                    chunk_embeddings.append(response.data[0].embedding)
                except Exception as e:
                    print(f"Error with chunk of length {len(chunk)}: {e}")
                    # If still too long, split in half and retry
                    half = len(chunk) // 2
                    for sub_chunk in [chunk[:half], chunk[half:]]:
                        response = openai_client.embeddings.create(
                            input=[sub_chunk],
                            model=model
                        )
                        chunk_embeddings.append(response.data[0].embedding)

            # Average the chunk embeddings (equal weight per chunk)
            avg_embedding = np.mean(chunk_embeddings, axis=0)
            embeddings.append(avg_embedding)
        else:
            # Process normally for short texts
            try:
                response = openai_client.embeddings.create(
                    input=[text],
                    model=model
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error with text of length {len(text)}: {e}")
                # If error, chunk it anyway
                chunks = chunk_text(text, max_tokens=6000)
                chunk_embeddings = []
                for chunk in chunks:
                    response = openai_client.embeddings.create(
                        input=[chunk],
                        model=model
                    )
                    chunk_embeddings.append(response.data[0].embedding)
                avg_embedding = np.mean(chunk_embeddings, axis=0)
                embeddings.append(avg_embedding)

    return np.array(embeddings)


# Check if embeddings already exist
embeddings_dir = 'embeddings'
os.makedirs(embeddings_dir, exist_ok=True)

conversation_embeddings_path = os.path.join(embeddings_dir, 'conversation_embeddings.pkl')
output_embeddings_path = os.path.join(embeddings_dir, 'output_embeddings.pkl')

if os.path.exists(conversation_embeddings_path):
    print("Loading existing embeddings...")
    with open(conversation_embeddings_path, 'rb') as f:
        conversation_embeddings = pickle.load(f)
else:
    print("Generating new embeddings...")
    # Get embeddings for conversation texts
    conversation_texts = df_combined['conversation_text'].tolist()
    conversation_embeddings = get_embeddings(conversation_texts)

    # Save embeddings
    with open(conversation_embeddings_path, 'wb') as f:
        pickle.dump(conversation_embeddings, f)


if os.path.exists(output_embeddings_path):
    print("Loading existing embeddings...")
    with open(output_embeddings_path, 'rb') as f:
        output_embeddings = pickle.load(f)
else:
    print("Generating new embeddings...")

    # Get embeddings for output texts: cache so same texts stay close in UMAP space
    output_texts = df_combined['output_text'].tolist()
    output_texts = [txt.lower() for txt in output_texts]
    output_embeddings = get_embeddings_cached(output_texts)

    with open(output_embeddings_path, 'wb') as f:
        pickle.dump(output_embeddings, f)


print(f"Conversation embeddings shape: {conversation_embeddings.shape}")
print(f"Output embeddings shape: {output_embeddings.shape}")

print("\nDone!")
# Step 5: Reduce to 2D using UMAP
print("\nStep 5: Reducing to 2D using UMAP...")

embeddings_2d_path = os.path.join(embeddings_dir, 'embeddings_2d.pkl')

if os.path.exists(embeddings_2d_path):
    print("Loading existing 2D embeddings...")
    with open(embeddings_2d_path, 'rb') as f:
        embeddings_2d = pickle.load(f)

else:

    # Combine all embeddings for joint UMAP transformation
    all_embeddings = np.vstack([conversation_embeddings, output_embeddings])

    # UMAP parameters optimized for preserving directional relationships
    umap_model = UMAP(
        n_components=2,
        n_neighbors=30,  # Balance between local and global structure
        min_dist=0.1,    # Allow some spread to see patterns
        metric='cosine', # Good for text embeddings
        random_state=42,
        n_epochs=500,    # More epochs for better convergence
        init='spectral', # Better initialization for structure preservation
        spread=1.5,      # Slightly increase spread for visibility
        repulsion_strength=1.0,  # Standard repulsion
        negative_sample_rate=5,  # Standard negative sampling
        transform_queue_size=4.0,  # Default
        a=None,  # Will be set automatically based on spread/min_dist
        b=None,  # Will be set automatically based on spread/min_dist
        local_connectivity=1.0,  # Ensure local connectivity
        set_op_mix_ratio=1.0,  # Pure fuzzy union
        target_n_neighbors=-1,  # Not doing supervised
        target_metric='categorical',
        target_weight=0.5,
        transform_seed=42,
        angular_rp_forest=False,  # Use NN-descent instead
        verbose=True
    )

    # Fit and transform
    print("Fitting UMAP model...")
    embeddings_2d = umap_model.fit_transform(all_embeddings)

    # Save as .npy file
    with open(embeddings_2d_path, 'wb') as f:
        pickle.dump(embeddings_2d, f)

    print(f"2D embeddings saved to: {embeddings_2d_path}")


# Split back into conversation and output embeddings
n_samples = len(df_combined)
conversation_2d = embeddings_2d[:n_samples]
output_2d = embeddings_2d[n_samples:]


# Add this to the end of your original script, after loading embeddings

print("\n" + "="*60)
print("CHECKING FOR EMBEDDING INCONSISTENCIES")
print("="*60)

# Check your saved embeddings for the known duplicates
duplicate_texts = ["Curious playful spark", "curious creative spark", "Curious creative spark", "Creative Flow Surge", "playful curiosity", "Ethical Guardian Core"]

for text in duplicate_texts:
    # Find all indices where this text appears
    indices = df_combined[df_combined['output_text'] == text].index.tolist()
    if len(indices) > 1:
        embeddings_for_text = output_embeddings[indices]
        print(f"\nText: '{text}' appears {len(indices)} times at indices {indices}")

        # Compare all pairs of embeddings for this text
        for i in range(len(embeddings_for_text)):
            for j in range(i+1, len(embeddings_for_text)):
                identical = np.array_equal(embeddings_for_text[i], embeddings_for_text[j])
                max_diff = np.max(np.abs(embeddings_for_text[i] - embeddings_for_text[j]))

                if not identical:
                    print(f"  Indices {indices[i]} vs {indices[j]}: Identical={identical}, Max diff={max_diff:.10f}")
    else:
        print(f"Text: '{text}' - only appears once")

print("\nDone checking!")



pd.DataFrame(conversation_2d).to_csv('embeddings/convo_2d.csv', index = False)
pd.DataFrame(output_2d).to_csv('embeddings/output_2d.csv', index = False)

# Step 6: Create magnetic force diagram
print("\nStep 6: Creating magnetic force diagram...")
plt.figure(figsize=(16, 12))

# Set style
sns.set_style("whitegrid", {'grid.alpha': 0.3})

# Calculate displacement vectors
displacements = output_2d - conversation_2d
distances = np.linalg.norm(displacements, axis=1)

# Normalize distances for coloring
distances_norm = (distances - distances.min()) / (distances.max() - distances.min())

# Create colormap
cmap = plt.cm.viridis

# Plot arrows with varying transparency based on distance
for i in range(n_samples):
    # Calculate alpha based on distance (longer arrows more transparent)
    alpha = 0.15 + 0.35 * (1 - distances_norm[i])  # Range from 0.15 to 0.5

    # Create arrow
    arrow = FancyArrowPatch(
        conversation_2d[i], output_2d[i],
        connectionstyle="arc3,rad=0.1",  # Slight curve for aesthetics
        arrowstyle='->,head_width=0.3,head_length=0.4',
        color=cmap(distances_norm[i]),
        alpha=alpha,
        linewidth=0.8,
        zorder=1
    )
    plt.gca().add_patch(arrow)

# Plot start and end points
plt.scatter(conversation_2d[:, 0], conversation_2d[:, 1],
           c='lightcoral', s=20, alpha=0.6, edgecolors='darkred',
           linewidth=0.5, label='Conversation Start', zorder=2)
plt.scatter(output_2d[:, 0], output_2d[:, 1],
           c='lightblue', s=20, alpha=0.6, edgecolors='darkblue',
           linewidth=0.5, label='AI Output End', zorder=3)

# Add colorbar for distance
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=distances.min(), vmax=distances.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label('Displacement Distance', rotation=270, labelpad=20)

# Labels and title
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.title('Conversation to AI Output: Embedding Space Movement Patterns\n(1000 conversation-output pairs)',
          fontsize=14, pad=20)
plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# Add grid
plt.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = 'embeddings/magnetic_force_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nDiagram saved to: {output_path}")

# Additional analysis: Calculate and print statistics
print("\n--- Movement Statistics ---")
print(f"Mean displacement distance: {distances.mean():.4f}")
print(f"Std displacement distance: {distances.std():.4f}")
print(f"Min displacement distance: {distances.min():.4f}")
print(f"Max displacement distance: {distances.max():.4f}")

# Find convergence points (areas where many outputs cluster)
from sklearn.cluster import DBSCAN

# Cluster the output points to find convergence areas
clustering = DBSCAN(eps=0.5, min_samples=10).fit(output_2d)
n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"\nNumber of convergence clusters found: {n_clusters}")

# Create a second plot showing density of end points
plt.figure(figsize=(12, 10))
plt.hexbin(output_2d[:, 0], output_2d[:, 1], gridsize=30, cmap='YlOrRd', alpha=0.8)
plt.colorbar(label='Number of AI outputs')
plt.scatter(conversation_2d[:, 0], conversation_2d[:, 1],
           c='blue', s=5, alpha=0.3, label='Conversation starts')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('Density of AI Output Positions in Embedding Space')
plt.legend()
plt.tight_layout()
plt.savefig('embeddings/output_density_map.png', dpi=300, bbox_inches='tight')
print("\nDensity map saved to: embeddings/output_density_map.png")

plt.show()



import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale

# Step 6: Create interactive magnetic force diagram
print("\nStep 6: Creating interactive magnetic force diagram...")

# Calculate displacement vectors
displacements = output_2d - conversation_2d
distances = np.linalg.norm(displacements, axis=1)

# Normalize distances for coloring
distances_norm = (distances - distances.min()) / (distances.max() - distances.min())

# Create subplot with secondary y-axis for the text panel
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    subplot_titles=["Conversation to AI Output Movement", "Selected Texts"]
)

# Sample colorscale for arrows (equivalent to viridis)
colors = sample_colorscale('viridis', distances_norm.tolist())

# Add arrows as line segments with arrowheads
for i in range(n_samples):
    # Add arrow line
    fig.add_trace(
        go.Scatter(
            x=[conversation_2d[i, 0], output_2d[i, 0]],
            y=[conversation_2d[i, 1], output_2d[i, 1]],
            mode="lines",
            line=dict(
                color=colors[i],
                width=1,
            ),
            opacity=0.15 + 0.35 * (1 - distances_norm[i]),
            showlegend=False,
            hoverinfo='skip',
            name=f"Arrow_{i}"
        ),
        row=1, col=1
    )

    # Add arrowhead at the end point
    # Calculate arrow direction
    dx = output_2d[i, 0] - conversation_2d[i, 0]
    dy = output_2d[i, 1] - conversation_2d[i, 1]
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:  # Avoid division by zero
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length

        # Create arrowhead (small triangle)
        arrow_size = 0.1
        perpendicular_x = -dy_norm * arrow_size * 0.5
        perpendicular_y = dx_norm * arrow_size * 0.5

        arrow_x = [
            output_2d[i, 0] - dx_norm * arrow_size + perpendicular_x,
            output_2d[i, 0],
            output_2d[i, 0] - dx_norm * arrow_size - perpendicular_x,
            output_2d[i, 0] - dx_norm * arrow_size + perpendicular_x
        ]
        arrow_y = [
            output_2d[i, 1] - dy_norm * arrow_size + perpendicular_y,
            output_2d[i, 1],
            output_2d[i, 1] - dy_norm * arrow_size - perpendicular_y,
            output_2d[i, 1] - dy_norm * arrow_size + perpendicular_y
        ]

        fig.add_trace(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode="lines",
                fill="toself",
                line=dict(color=colors[i], width=0),
                fillcolor=colors[i],
                opacity=0.15 + 0.35 * (1 - distances_norm[i]),
                showlegend=False,
                hoverinfo='skip',
                name=f"Arrowhead_{i}"
            ),
            row=1, col=1
        )

# Add conversation start points
conversation_hover_text = [
    f"Index: {i}<br>Conversation: {df_combined.iloc[i]['conversation_text'][:100]}..."
    for i in range(n_samples)
]

fig.add_trace(
    go.Scatter(
        x=conversation_2d[:, 0],
        y=conversation_2d[:, 1],
        mode="markers",
        marker=dict(
            color='lightcoral',
            size=6,
            line=dict(color='darkred', width=0.5),
            opacity=0.8
        ),
        name="Conversation Start",
        hovertext=conversation_hover_text,
        hovertemplate="<b>Conversation Start</b><br>%{hovertext}<extra></extra>",
        customdata=list(range(n_samples)),  # Store indices for selection
        selectedpoints=[],  # Initialize empty selection
    ),
    row=1, col=1
)

# Add AI output end points
output_hover_text = [
    f"Index: {i}<br>Output: {df_combined.iloc[i]['output_text'][:100]}..."
    for i in range(n_samples)
]

fig.add_trace(
    go.Scatter(
        x=output_2d[:, 0],
        y=output_2d[:, 1],
        mode="markers",
        marker=dict(
            color='lightblue',
            size=6,
            line=dict(color='darkblue', width=0.5),
            opacity=0.8
        ),
        name="AI Output End",
        hovertext=output_hover_text,
        hovertemplate="<b>AI Output End</b><br>%{hovertext}<extra></extra>",
        customdata=list(range(n_samples)),  # Store indices for selection
        selectedpoints=[],  # Initialize empty selection
    ),
    row=1, col=1
)

# Add empty trace for text display
fig.add_trace(
    go.Scatter(
        x=[],
        y=[],
        mode="text",
        text=[],
        textposition="middle left",
        showlegend=False,
        name="Selected Texts"
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title={
        'text': "Interactive Conversation to AI Output: Embedding Space Movement Patterns<br><sub>Select points by drawing a box around them</sub>",
        'x': 0.5,
        'xanchor': 'center'
    },
    height=800,
    width=1400,
    dragmode="select",  # Enable box select
    selectdirection="d",
    hovermode="closest",
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Update x and y axes for the main plot
fig.update_xaxes(title_text="UMAP Dimension 1", row=1, col=1)
fig.update_yaxes(title_text="UMAP Dimension 2", row=1, col=1)

# Hide axes for text panel
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

# Add JavaScript callback for selection handling
fig.update_layout(
    selectdirection="d"
)

# Create custom callback using plotly.graph_objects
callback_script = """
<script>
function updateTextPanel() {
    // This function would be called when selection changes
    // For now, we'll handle this through the Plotly event system
    console.log('Selection updated');
}

// Add event listener for selection
document.addEventListener('DOMContentLoaded', function() {
    var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
    if (plotDiv) {
        plotDiv.on('plotly_selected', function(eventData) {
            if (eventData && eventData.points) {
                var selectedTexts = [];
                var yPosition = 0.9;

                eventData.points.forEach(function(point, index) {
                    if (point.customdata !== undefined) {
                        var idx = point.customdata;
                        var text = '';

                        if (point.fullData.name === 'Conversation Start') {
                            text = 'Conv ' + idx + ': ' + point.hovertext.split('Conversation: ')[1].split('...')[0] + '...';
                        } else if (point.fullData.name === 'AI Output End') {
                            text = 'Out ' + idx + ': ' + point.hovertext.split('Output: ')[1].split('...')[0] + '...';
                        }

                        if (text) {
                            selectedTexts.push({
                                x: 0.02,
                                y: yPosition,
                                text: text,
                                font: {size: 10}
                            });
                            yPosition -= 0.05;
                        }
                    }
                });

                // Update the text trace
                Plotly.restyle(plotDiv, {
                    'x': [selectedTexts.map(t => t.x)],
                    'y': [selectedTexts.map(t => t.y)],
                    'text': [selectedTexts.map(t => t.text)]
                }, [plotDiv.data.length - 1]);
            }
        });
    }
});
</script>
"""

# For Jupyter notebooks, we'll create a version that works with widgets
try:
    from IPython.display import HTML, display

    # Create the plot
    fig.show()

    # Add custom JavaScript for handling selections
    display(HTML(callback_script))

except ImportError:
    # For standalone Python, just show the plot
    fig.show()

# Save the interactive plot
fig.write_html("embeddings/interactive_magnetic_force_diagram.html")
print(f"\nInteractive diagram saved to: embeddings/interactive_magnetic_force_diagram.html")
