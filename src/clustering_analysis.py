import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', cache_folder='cache'):
    """
    Generates embeddings for a list of texts using a sentence-transformer model.
    
    Args:
        texts (list): A list of strings to be embedded.
        model_name (str): The name of the sentence-transformer model to use.
        cache_folder (str): Folder to cache the downloaded model.

    Returns:
        np.ndarray: An array of embeddings.
    """
    print(f"Generating embeddings using '{model_name}'...")
    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    embeddings = model.encode(texts, show_progress_bar=True)
    print("Embeddings generated successfully.")
    return embeddings

def find_optimal_k(data, max_k=10, viz_path='outputs/visualizations/elbow_plot.png'):
    """
    Finds the optimal number of clusters (k) using the Elbow Method.
    
    Args:
        data (np.ndarray): The data to cluster.
        max_k (int): The maximum number of clusters to test.
        viz_path (str): Path to save the elbow plot visualization.

    Returns:
        int: The optimal k value.
    """
    print("Finding optimal k using the Elbow Method...")
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)

    # Plot the elbow
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(viz_path)
    plt.close() 
    
    print(f"Please inspect '{viz_path}' to determine the optimal k.")
    # For this script, we will proceed with a default, but inspection is recommended.
    return 5 # Placeholder, should be determined from the plot

def perform_clustering(embeddings, n_clusters):
    """
    Performs K-Means clustering on the embeddings.
    
    Args:
        embeddings (np.ndarray): The embeddings to cluster.
        n_clusters (int): The number of clusters to form.

    Returns:
        KMeans: The fitted KMeans model object.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    print("Clustering complete.")
    return kmeans

def visualize_clusters_2d(embeddings, labels, file_path, title='t-SNE 2D Visualization of Clusters'):
    """
    Visualizes clusters in 2D using t-SNE.
    """
    print(f"Creating 2D t-SNE visualization at {file_path}...")
    # Set perplexity to be less than the number of samples
    perplexity = min(30, len(embeddings) - 1)
    if perplexity <= 0:
        print("Warning: Cannot create t-SNE plot with 1 or fewer samples. Skipping.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(len(set(labels))))
    plt.savefig(file_path)
    plt.close()
    print(f"2D visualization saved.")

def visualize_clusters_3d(embeddings, labels, file_path, title='Interactive 3D Visualization of Clusters'):
    """
    Visualizes clusters in 3D using PCA for dimensionality reduction.
    """
    print(f"Creating 3D interactive visualization at {file_path}...")
    if embeddings.shape[1] < 3:
        print("Warning: Cannot create 3D plot with fewer than 3 dimensions. Skipping.")
        return

    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(file_path)
    print(f"3D visualization saved.")


if __name__ == '__main__':
    # Define directory paths to match your project structure
    output_dir = 'outputs'
    viz_dir = os.path.join(output_dir, 'visualizations')
    
    # Create directories if they don't exist
    os.makedirs(viz_dir, exist_ok=True)

    # Define input file path
    staged_data_path = os.path.join(output_dir, 'stage_conversations.csv')

    # Load the segmented conversation data
    try:
        staged_df = pd.read_csv(staged_data_path)
    except FileNotFoundError:
        print(f"Error: '{staged_data_path}' not found. Please run 'stage_segmentation.py' first.")
        exit()

    if staged_df.empty:
        print("The 'stage_conversations.csv' file is empty. Cannot proceed with clustering.")
        exit()

    # --- Method A: Without Fan Profiling ---
    print("\n--- Starting Method A: Clustering without Fan Profiles ---")
    texts_to_embed = staged_df['text'].astype(str).fillna('').tolist()
    embeddings_without_profile = generate_embeddings(texts_to_embed)
    
    # Save the embeddings to the outputs folder
    embeddings_a_path = os.path.join(output_dir, 'embeddings_without_profile.pkl')
    with open(embeddings_a_path, 'wb') as f:
        pickle.dump(embeddings_without_profile, f)
    print(f"✅ Method A embeddings saved to '{embeddings_a_path}'")

    # Find optimal k and perform clustering
    elbow_plot_a_path = os.path.join(viz_dir, 'elbow_plot_method_a.png')
    optimal_k_a = find_optimal_k(embeddings_without_profile, viz_path=elbow_plot_a_path)
    kmeans_a = perform_clustering(embeddings_without_profile, optimal_k_a)
    staged_df['cluster_a'] = kmeans_a.labels_

    # Visualize the clusters
    viz_2d_a_path = os.path.join(viz_dir, 'method_a_2d_tsne.png')
    viz_3d_a_path = os.path.join(viz_dir, 'method_a_3d_pca.html')
    visualize_clusters_2d(embeddings_without_profile, kmeans_a.labels_, file_path=viz_2d_a_path, title='Method A - 2D t-SNE Visualization')
    visualize_clusters_3d(embeddings_without_profile, kmeans_a.labels_, file_path=viz_3d_a_path, title='Method A - 3D Interactive Visualization')

    # --- Method B: With Fan Profiling (Conceptual) ---
    print("\n--- Starting Method B: Clustering with Fan Profiles (Conceptual) ---")
    # This is a conceptual placeholder. It requires fan profile features from Phase 2.
    # We will generate random features for demonstration purposes.
    
    # Create placeholder profile features
    num_profiles = staged_df['fan_model_id'].nunique()
    unique_fan_models = staged_df['fan_model_id'].unique()
    placeholder_profiles = pd.DataFrame({
        'fan_model_id': unique_fan_models,
        'profile_feature1': np.random.rand(num_profiles),
        'profile_feature2': np.random.rand(num_profiles)
    })
    
    # Merge profile features with staged data
    staged_df_with_profiles = pd.merge(staged_df, placeholder_profiles, on='fan_model_id', how='left').fillna(0)
    profile_features = staged_df_with_profiles[['profile_feature1', 'profile_feature2']].values
    
    # Combine text embeddings with profile features
    embeddings_with_profile = np.concatenate((embeddings_without_profile, profile_features), axis=1)
    
    # Save the combined embeddings to the outputs folder
    embeddings_b_path = os.path.join(output_dir, 'embeddings_with_profile.pkl')
    with open(embeddings_b_path, 'wb') as f:
        pickle.dump(embeddings_with_profile, f)
    print(f"✅ Method B embeddings saved to '{embeddings_b_path}'")

    # Find optimal k and perform clustering
    elbow_plot_b_path = os.path.join(viz_dir, 'elbow_plot_method_b.png')
    optimal_k_b = find_optimal_k(embeddings_with_profile, viz_path=elbow_plot_b_path)
    kmeans_b = perform_clustering(embeddings_with_profile, optimal_k_b)
    staged_df['cluster_b'] = kmeans_b.labels_

    # Visualize the new clusters
    viz_2d_b_path = os.path.join(viz_dir, 'method_b_2d_tsne.png')
    viz_3d_b_path = os.path.join(viz_dir, 'method_b_3d_pca.html')
    visualize_clusters_2d(embeddings_with_profile, kmeans_b.labels_, file_path=viz_2d_b_path, title='Method B - 2D t-SNE Visualization')
    visualize_clusters_3d(embeddings_with_profile, kmeans_b.labels_, file_path=viz_3d_b_path, title='Method B - 3D Interactive Visualization')

    # Save the final dataframe with cluster labels to the outputs folder
    final_csv_path = os.path.join(output_dir, 'clustered_stages.csv')
    staged_df.to_csv(final_csv_path, index=False)
    print(f"\n✅ Final data with cluster labels saved to '{final_csv_path}'")