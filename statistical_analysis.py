import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import skimage.morphology as morpho
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
import SimpleITK as sitk
from collections import defaultdict


dataset_dir='./data/'

ndata=1000
n_extrema=20
n_clusters=1
alpha=0.5
fontsize=30

centering=False

scalar_field_target=np.load(dataset_dir+'phi_10000_32^3.npy')[0:ndata]
scalar_value_target=np.squeeze(np.load(dataset_dir+'tough_10000_32^3.npy')[0:ndata],axis=(2,3,4))

## Set matplotlib font
plt.rcParams['font.family'] = 'serif'
## Set axes and colorbar font size
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize


def standardize_toughness(toughness):
    """Standardize the toughness values to have mean 0 and std 1."""
    mean_toughness = np.mean(scalar_value_target)
    std_toughness = np.std(scalar_value_target)
    standardized_toughness = (toughness - mean_toughness) / std_toughness
    return standardized_toughness


def watershed_3d_sitk(image_3d):
    sitk_image = sitk.GetImageFromArray(image_3d)
    watershed_filter = sitk.MorphologicalWatershedImageFilter()
    watershed_filter.SetLevel(0.11)
    watershed_filter.SetMarkWatershedLine(False)
    labeled_image = watershed_filter.Execute(sitk_image)
    return sitk.GetArrayFromImage(labeled_image)

def count_number_of_voxels_in_each_region(image):
    labels = watershed_3d_sitk(image)

    # Count the number of voxels in each region
    unique, counts = np.unique(labels, return_counts=True)
    region_counts = dict(zip(unique, counts))

    return region_counts


def plot_3d_image(img, title, ax):
    """Plot orthogonal slices of a 3D image"""

    # Calculate middle slices
    z_slice = img[img.shape[0]//2, :, :]
    y_slice = img[:, img.shape[1]//2, :]
    x_slice = img[:, :, img.shape[2]//2]
    
    # Plot slices
    ax[0].imshow(z_slice, cmap='viridis')
    ax[0].set_title('Axial View (Z)')
    ax[0].axis('off')
    
    ax[1].imshow(y_slice, cmap='viridis')
    ax[1].set_title('Coronal View (Y)')
    ax[1].axis('off')
    
    ax[2].imshow(x_slice, cmap='viridis')
    ax[2].set_title('Sagittal View (X)')
    ax[2].axis('off')
    
    plt.suptitle(title)


def cluster_image_batch(image_array, output_dir, n_clusters=2):
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of 3D images
    images = [image_array[f,0] for f in range(len(image_array))]

    # Feature extraction
    features = np.array([[
        # img.flatten()
        np.percentile(img, 99), 
        np.std(img),
        np.mean(img),
    ] for img in images])
    
    # Clustering with n_clusters
    scaled_features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # PCA for visualization
    pca = PCA(n_components=3)
    pc_components = pca.fit_transform(scaled_features)
  
    # Create main plot
    fig = plt.figure(figsize=(15, 8))
    ax_main = fig.add_subplot(111)

    ## Create a colormap using toughness values as the limits of the color scale
    colormap = plt.cm.get_cmap('RdYlBu', 100)  # Use a diverging colormap
    norm = plt.Normalize(vmin=np.min(standardize_toughness(scalar_value_target[:,0])), 
                         vmax=np.max(standardize_toughness(scalar_value_target[:,0])))


    scatter = ax_main.scatter(pc_components[:, 0], pc_components[:, 1], 
                            c=standardize_toughness(scalar_value_target[:,0]), cmap=colormap, s=50, alpha=alpha)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label(r'$\frac{G - \mu_G}{\sigma_G}$', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=20)
    
    
    # Plot cluster centers in PCA space
    centers_pca = pca.transform(kmeans.cluster_centers_)
    
    ax_main.set_xlabel('PC1',  fontsize=fontsize)
    ax_main.set_ylabel('PC2',  fontsize=fontsize)

    ax_main.tick_params(axis='both', which='major', labelsize=20)
    ax_main.tick_params(axis='both', which='minor', labelsize=20)

    # Find and plot nearest actual images to cluster centers
    centroid_indices = []
    for cluster_id in range(n_clusters):
        # Find closest image to cluster center
        cluster_mask = clusters == cluster_id
        cluster_features = scaled_features[cluster_mask]
        
        if len(cluster_features) > 0:  # Check for empty clusters
            distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[cluster_id], axis=1)
            centroid_idx = np.where(cluster_mask)[0][np.argmin(distances)]
            centroid_indices.append(centroid_idx)

    # Create figure for center examples
    fig_centers = plt.figure(figsize=(n_clusters*5, 5))
    
    # Plot center examples
    for idx, cluster_id in enumerate(range(n_clusters)):
        if idx < len(centroid_indices):
            img_idx = centroid_indices[idx]
            img = images[img_idx]
            
            # Create subplot for 3 orthogonal views
            ax = fig_centers.add_subplot(1, n_clusters, idx+1)
            
            # Plot middle slices
            y_slice = img[:, img.shape[1]//2, :]
            x_slice = img[:, :, img.shape[2]//2]
            
            # Combine slices into single image
            combined = np.hstack([x_slice, y_slice])
            
            ax.imshow(combined, cmap='viridis')
            ax.set_title(f'Cluster {cluster_id} Center\nImage {img_idx} toughness {np.round(standardize_toughness(scalar_value_target[img_idx,0]),1)}',
                          fontdict={'fontsize': 20})
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_centers.pdf'), dpi=150)
    plt.close()

    # Plot cluster maxima
    # Find and plot the furthest images from cluster centers
    extreme_indices = []

    for cluster_id in range(n_clusters):
        # Find closest image to cluster center
        cluster_mask = clusters == cluster_id
        cluster_features = scaled_features[cluster_mask]
        
        if len(cluster_features) > 0:  # Check for empty clusters
            distances = np.abs(standardize_toughness(scalar_value_target[:,0]))
            for index in range(2*n_extrema):
                # Get the index of the furthest point
                extreme_idx = np.where(cluster_mask)[0][np.argmax(distances)]
                # Remove the furthest point from the distances
                distances[extreme_idx] = -np.inf
                # Append the extreme index
                if extreme_idx not in extreme_indices:
                # Ensure we don't add duplicates
                # Append the extreme index
                    extreme_idx = np.where(cluster_mask)[0][np.argmax(distances)]
                    extreme_indices.append(extreme_idx)            
        
    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatterplot.pdf'), dpi=150)
    plt.close()


    # Create figure for extreme examples
    fig_extremes = plt.figure(figsize=(n_clusters*5, n_extrema*5))    

    # Plot all extreme examples for each cluster
    for idx, extreme_idx in enumerate(extreme_indices):
        if idx < len(extreme_indices):
            img = images[extreme_idx]
            
            # Create subplot for orthogonal views
            ax = fig_extremes.add_subplot(n_extrema, n_clusters, idx+1)
            
            # Plot middle slices
            y_slice = img[:, img.shape[1]//2, :]
            x_slice = img[:, :, img.shape[2]//2]
            
            # Combine slices into single image
            combined = np.hstack([x_slice, y_slice])
            
            ax.imshow(combined, cmap='viridis')
            ax.set_title(f'Extreme {idx+1}\nImage {extreme_idx} toughness {standardize_toughness(scalar_value_target[extreme_idx,0])}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_examples.pdf'), dpi=150)
    plt.close()


    # Save 3D visualizations of cluster centers
    for cluster_id in range(n_clusters):
        if cluster_id < len(centroid_indices):
            img_idx = centroid_indices[cluster_id]
            img = images[img_idx]
            
            fig3d = plt.figure(figsize=(8, 6))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            threshold = np.percentile(img, 75)
            mask = img > threshold
            
            ax3d.voxels(mask, facecolors=plt.cm.tab10(cluster_id), 
                       edgecolor=None, alpha=0.4)
            
            ax3d.set_title(f'Cluster {cluster_id} Center Structure ')
            plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_3d_center.pdf'), dpi=150)
            plt.close()

    # Plot PC1 vs standardized toughness
    fig_pc1 = plt.figure(figsize=(8, 6))
    ax_pc1 = fig_pc1.add_subplot(111) 
    scatter = ax_pc1.scatter(pc_components[:, 0], standardize_toughness(scalar_value_target[:,0]), 
                            c=clusters, cmap='tab10', s=50, alpha=alpha)
    ax_pc1.set_xlabel('PC1', fontsize=fontsize)
    ax_pc1.set_ylabel(r'$\frac{G - \mu_G}{\sigma_G}$', fontsize=fontsize)
    ax_pc1.tick_params(axis='both', which='major', labelsize=20)
    ax_pc1.tick_params(axis='both', which='minor', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pc1_vs_toughness.pdf'), dpi=150)
    plt.close()

    # Save PC1 and tough to file named pc1_vs_tough.txt as two columns
    pc1_tough = np.vstack((pc_components[:, 0], standardize_toughness(scalar_value_target[:,0]))).T
    np.savetxt(os.path.join(output_dir, 'pc1_vs_tough.txt'), pc1_tough, header='PC1 Standardized_Toughness')

    return pc_components, clusters

# Generate test images with more distinct clusters
def create_test_images(output_dir, n_images=20):
    Path(output_dir).mkdir(exist_ok=True)
    
    for i in range(n_images):
        if i % 2 == 0:
            # Cluster 1: High-intensity structures
            img = np.random.normal(0.1, 0.05, (32, 32, 32))
            coords = np.indices((32, 32, 32)) - 16
            mask = np.sum(coords**2, axis=0) <= 25  # Spherical structure
            img[mask] += 0.5
        else:
            # Cluster 2: Low-intensity with random noise
            img = np.random.normal(0.4, 0.3, (32, 32, 32))
            
        np.save(Path(output_dir)/f"image_{i:03d}.npy", img)

def centralize_crack(img):

    voxels_in_region_dict=count_number_of_voxels_in_each_region(img)

    counter=0
    while len(voxels_in_region_dict)!=2:
        img=np.roll(img, 1, axis=0)
        voxels_in_region_dict=count_number_of_voxels_in_each_region(img)
        counter=counter+1
        if counter>20:
            raise Exception('Error: too many iterations')


    if voxels_in_region_dict[1]<500 or voxels_in_region_dict[2]<500:
        img=np.roll(img, 3, axis=0)
        voxels_in_region_dict=count_number_of_voxels_in_each_region(img)

    if voxels_in_region_dict[1] < voxels_in_region_dict[2]:
        idx_menor_og='1'
    else:
        idx_menor_og='2'

    idx_menor=idx_menor_og
    counter=0
    while idx_menor==idx_menor_og:
        if idx_menor=='1':
            rolled=np.roll(img, 1, axis=0)
            voxels_in_region_dict=count_number_of_voxels_in_each_region(rolled)
            if len(voxels_in_region_dict)==2:
                if voxels_in_region_dict[1] < voxels_in_region_dict[2]:
                    idx_menor='1'
                else:
                    idx_menor='2'
            else:
                while len(voxels_in_region_dict)!=2:
                    rolled=np.roll(rolled, 1, axis=0)
                    voxels_in_region_dict=count_number_of_voxels_in_each_region(rolled)
                    counter=counter+1
                    if counter>20:
                        raise Exception('Error: too many iterations')
            img=rolled
        else:
            rolled=np.roll(img, -1, axis=0)
            voxels_in_region_dict=count_number_of_voxels_in_each_region(rolled)
            if len(voxels_in_region_dict)==2:
                if voxels_in_region_dict[1] < voxels_in_region_dict[2]:
                    idx_menor='1'
                else:
                    idx_menor='2'
            else:
                while len(voxels_in_region_dict)!=2:
                    rolled=np.roll(rolled, -1, axis=0)
                    voxels_in_region_dict=count_number_of_voxels_in_each_region(rolled)
                    counter=counter+1
                    if counter>20:
                        raise Exception('Error: too many iterations')
            img=rolled
        if counter>20:
            raise Exception('Error: too many iterations')

    return rolled

if __name__ == "__main__":

    if centering==True:
        for idx in range(len(scalar_field_target)):
            img=scalar_field_target[idx,0]
            try:
                img=centralize_crack(img)
            except:
                print('error')
                continue
            scalar_field_target[idx,0]=img

    if ndata>=3975:
        scalar_field_target=np.delete(scalar_field_target,3974,axis=0)

    # Run clustering and visualization
    cluster_image_batch(
        image_array=scalar_field_target,
        output_dir="cluster_results",
        n_clusters=n_clusters,  # Specify desired number of clusters
    )