import jittor as jt
import numpy as np
import os
import random
from jittor.dataset import Dataset

import os
from typing import List, Dict, Callable, Union

from .asset import Asset
from .sampler import Sampler, SamplerMix
from .format import id_to_name, num_joints

# Define hand joint IDs (9 and 13 are left and right wrists, 22-51 are finger joints)
HAND_JOINT_IDS = [9, 13] + list(range(22, 52))

class HandAwareSampler(SamplerMix):
    """
    Hand-aware sampler that provides higher sampling density for hand regions
    """
    def __init__(self, num_samples: int, vertex_samples: int, hand_boost_ratio: float = 3.0):
        super().__init__(num_samples, vertex_samples)
        self.hand_boost_ratio = hand_boost_ratio
    
    def sample(
        self,
        vertices: np.ndarray,
        vertex_normals: np.ndarray,
        face_normals: np.ndarray,
        vertex_groups: dict[str, np.ndarray],
        faces: np.ndarray,
    ):
        """
        Perform high-density sampling for hand regions
        """
        if self.num_samples == -1:
            return vertices, vertex_normals, vertex_groups
        
        # If skin weight information exists, identify hand vertices
        if 'skin' in vertex_groups and vertex_groups['skin'].shape[1] >= 52:
            skin_weights = vertex_groups['skin']
            
            # Calculate association strength between each vertex and hand joints
            hand_weights = np.sum(skin_weights[:, HAND_JOINT_IDS], axis=1)
            hand_vertices_mask = hand_weights > 0.1  # Adjustable threshold
            
            # If hand vertices found, perform stratified sampling
            if np.any(hand_vertices_mask):
                hand_vertex_count = np.sum(hand_vertices_mask)
                body_vertex_count = len(vertices) - hand_vertex_count
                
                # Allocate more sampling points for hands
                total_vertex_samples = min(self.vertex_samples, len(vertices))
                hand_vertex_samples = min(
                    int(total_vertex_samples * 0.4),  # 40% sampling points for hands
                    hand_vertex_count
                )
                body_vertex_samples = total_vertex_samples - hand_vertex_samples
                
                # Separately sample hand and body vertices
                hand_indices = np.where(hand_vertices_mask)[0]
                body_indices = np.where(~hand_vertices_mask)[0]
                
                selected_hand = np.random.choice(hand_indices, hand_vertex_samples, replace=False)
                selected_body = np.random.choice(body_indices, min(body_vertex_samples, len(body_indices)), replace=False)
                
                selected_vertices = np.concatenate([selected_hand, selected_body])
                
                # Build sampling results
                n_vertices = vertices[selected_vertices]
                n_normal = vertex_normals[selected_vertices]
                n_v = {name: v[selected_vertices] for name, v in vertex_groups.items()}
            else:
                # No hand vertices, use default sampling
                perm = np.random.permutation(vertices.shape[0])
                vertex_samples = min(self.vertex_samples, vertices.shape[0])
                perm = perm[:vertex_samples]
                n_vertices = vertices[perm]
                n_normal = vertex_normals[perm]
                n_v = {name: v[perm] for name, v in vertex_groups.items()}
        else:
            # No skin information, use default sampling
            perm = np.random.permutation(vertices.shape[0])
            vertex_samples = min(self.vertex_samples, vertices.shape[0])
            perm = perm[:vertex_samples]
            n_vertices = vertices[perm]
            n_normal = vertex_normals[perm]
            n_v = {name: v[perm] for name, v in vertex_groups.items()}
        
        # 2. Surface sampling (same as original)
        num_surface_samples = self.num_samples - len(n_vertices)
        if num_surface_samples > 0:
            vertex_samples, face_index, random_lengths = sample_surface(
                num_samples=num_surface_samples,
                vertices=vertices,
                faces=faces,
                return_weight=True,
            )
            vertex_samples = np.concatenate([n_vertices, vertex_samples], axis=0)
            normal_samples = np.concatenate([n_normal, face_normals[face_index]], axis=0)
            vertex_groups_samples = {}
            for n, v in vertex_groups.items():
                g = self._sample_barycentric(
                    vertex_groups=v,
                    faces=faces,
                    face_index=face_index,
                    random_lengths=random_lengths,
                )
                vertex_groups_samples[n] = np.concatenate([n_v[n], g], axis=0)
        else:
            vertex_samples = n_vertices
            normal_samples = n_normal
            vertex_groups_samples = n_v
            
        return vertex_samples, normal_samples, vertex_groups_samples

def sample_surface(
    num_samples: int,
    vertices: np.ndarray,
    faces: np.ndarray,
    return_weight: bool=False,
):
    '''
    Randomly pick samples according to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    '''
    # get face area
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    face_weight = (face_weight * face_weight).sum(axis=1)
    
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    
    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    if not return_weight:
        return vertex_samples
    return vertex_samples, face_index, random_lengths

def transform(asset: Asset):
    """
    Transform the asset data into [-1, 1]^3.
    """
    # Find min and max values for each dimension of points
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Calculate the scale factor to normalize to [-1, 1]
    # We take the maximum range across all dimensions to preserve aspect ratio
    scale = np.max(max_vals - min_vals) / 2
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (asset.vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        normalized_joints = (asset.joints - center) / scale
    else:
        normalized_joints = None
    
    asset.vertices  = normalized_vertices
    asset.joints    = normalized_joints
    # remember to change matrix_local !
    if asset.matrix_local is not None:
        asset.matrix_local[:, :3, 3] = normalized_joints

class RigDataset(Dataset):
    '''
    A simple dataset class with matrix_basis data augmentation.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False,
        use_pose_augmentation: bool = True,
        pose_augmentation_ratio: float = 0.8,
        random_pose_angle: float = 30.0,
        preload_data: bool = True
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # do not use `sampler` to avoid name conflict
        self.transform  = transform
        
        self.return_origin_vertices = return_origin_vertices
        self.use_pose_augmentation = use_pose_augmentation and train
        self.pose_augmentation_ratio = pose_augmentation_ratio
        self.random_pose_angle = random_pose_angle
        self.preload_data = preload_data
        
        # Preloaded data cache
        self.preloaded_assets = None
        
        # Preload matrix_basis from track files for data augmentation
        self.track_matrix_basis = []
        if self.use_pose_augmentation:
            self._load_track_matrix_basis()
        
        # Preload data to memory
        if self.preload_data:
            self._preload_all_data()
        
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def _load_track_matrix_basis(self):
        """
        Load matrix_basis from official track files for data augmentation
        """
        track_dir = os.path.join(self.data_root, 'track')
        if os.path.exists(track_dir):
            for filename in os.listdir(track_dir):
                if filename.endswith('.npz'):
                    try:
                        track_path = os.path.join(track_dir, filename)
                        data = np.load(track_path)
                        if 'matrix_basis' in data:
                            matrix_basis = data['matrix_basis']  # shape: (frame, J, 4, 4)
                            
                            # Traverse each frame
                            for frame_idx in range(matrix_basis.shape[0]):
                                frame_matrix = matrix_basis[frame_idx]  # shape: (J, 4, 4)
                                
                                # Extend to 52 joints (if original has only 22)
                                if frame_matrix.shape[0] == 22:
                                    # Create identity matrices for new finger joints
                                    extended_matrix = np.zeros((52, 4, 4))
                                    extended_matrix[:22] = frame_matrix
                                    # Use identity matrices or slight transformations for finger joints
                                    for i in range(22, 52):
                                        extended_matrix[i] = np.eye(4)
                                    frame_matrix = extended_matrix
                                
                                self.track_matrix_basis.append(frame_matrix)
                    except Exception as e:
                        print(f"Warning: Failed to load track file {filename}: {e}")
        
        print(f"Loaded {len(self.track_matrix_basis)} track matrix_basis frames for data augmentation")
    
    def _preload_all_data(self):
        """
        Preload all data to memory and pre-convert to jittor format to improve training speed
        """
        print(f"Preloading {len(self.paths)} assets to memory with format pre-conversion...")
        self.preloaded_data = []  # Store preprocessed data directly
        
        for i, path in enumerate(self.paths):
            try:
                asset = Asset.load(os.path.join(self.data_root, path))
                
                # Apply transform if provided
                original_asset = asset  # Save original asset for subsequent sampling
                if self.transform is not None:
                    self.transform(asset)
                
                # Pre-sample data
                sampled_asset = asset.sample(sampler=self._sampler)
                
                # Pre-convert to jittor format to avoid runtime conversion
                vertices = jt.array(sampled_asset.vertices, dtype=jt.float32)
                normals = jt.array(sampled_asset.normals, dtype=jt.float32)
                
                # Process joints
                if sampled_asset.joints is not None:
                    joints = jt.array(sampled_asset.joints, dtype=jt.float32)
                    # Ensure joint count is 52 and pre-pad
                    if joints.shape[0] < 52:
                        padded_joints = jt.zeros((52, 3), dtype=jt.float32)
                        padded_joints[:joints.shape[0]] = joints
                        joints = padded_joints
                else:
                    joints = None
                
                # Process skin weights
                if sampled_asset.skin is not None:
                    skin = jt.array(sampled_asset.skin, dtype=jt.float32)
                    # Ensure skin weight dimension is 52 and pre-pad
                    if skin.shape[1] < 52:
                        padded_skin = jt.zeros((skin.shape[0], 52), dtype=jt.float32)
                        padded_skin[:, :skin.shape[1]] = skin
                        skin = padded_skin
                else:
                    skin = None
                
                # Pre-compute common shape transformations to avoid runtime computation
                precomputed_data = {
                    'vertices': vertices,
                    'normals': normals,
                    'joints': joints,
                    'skin': skin,
                    'cls': asset.cls,
                    'id': asset.id,
                    # Pre-compute format for skeleton model - Fix dimensions
                    'vertices_permuted': vertices.transpose(0, 1) if vertices.ndim == 2 else None,  # [3, N] for old models
                    'joints_flattened': joints.view(-1) if joints is not None else None,  # [J*3] for old models
                    'joints_reshaped': joints.view(-1, 3) if joints is not None else None,  # [J, 3] for new models
                    # Pre-store original vertices for return_origin_vertices
                    'origin_vertices': jt.array(asset.vertices.copy(), dtype=jt.float32) if self.return_origin_vertices else None,
                    # Pre-compute common flags to avoid runtime checks
                    'joints_ndim': joints.ndim if joints is not None else 0,
                    'vertices_ndim': vertices.ndim,
                    # Save original numpy data for pose augmentation (avoid deep copying Asset)
                    'original_vertices': asset.vertices.copy() if self.use_pose_augmentation else None,
                    'original_joints': asset.joints.copy() if (self.use_pose_augmentation and asset.joints is not None) else None,
                    'original_matrix_local': asset.matrix_local.copy() if (self.use_pose_augmentation and hasattr(asset, 'matrix_local') and asset.matrix_local is not None) else None,
                }
                
                self.preloaded_data.append(precomputed_data)
                
                # Show progress
                if (i + 1) % 100 == 0 or (i + 1) == len(self.paths):
                    print(f"Preloaded and converted {i + 1}/{len(self.paths)} assets...")
                    
            except Exception as e:
                print(f"Warning: Failed to preload asset {path}: {e}")
                # Add None as placeholder
                self.preloaded_data.append(None)
        
        print(f"Data preloading completed. Loaded {len([d for d in self.preloaded_data if d is not None])}/{len(self.paths)} assets successfully.")
    
    def _apply_pose_augmentation(self, asset: Asset):
        """
        Apply pose augmentation, including track data and randomly generated poses
        60% probability for track augmentation, 40% probability for random augmentation
        """
        # 60% probability for track augmentation, 40% probability for random augmentation
        if random.random() < 0.6 and self.track_matrix_basis:
            # Use matrix_basis from track files (60% probability)
            matrix_basis = random.choice(self.track_matrix_basis).copy()
        else:
            # Generate random pose, use smaller angles for finger joints (40% probability)
            matrix_basis = asset.get_random_matrix_basis(self.random_pose_angle)
            # Use smaller random angles for finger joints
            hand_angle = self.random_pose_angle * 0.3  # Fingers use smaller changes
            for i in HAND_JOINT_IDS:
                if i < matrix_basis.shape[0]:
                    hand_matrix = asset.get_random_matrix_basis(hand_angle)
                    if i < hand_matrix.shape[0]:
                        matrix_basis[i] = hand_matrix[i]
        
        # Apply matrix_basis transformation
        try:
            asset.apply_matrix_basis(matrix_basis)
        except Exception as e:
            print(f"Warning: Failed to apply matrix_basis augmentation: {e}")
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            data (Dict): Dictionary containing the following keys:
                - vertices: jt.Var, (B, N, 3) point cloud data
                - normals: jt.Var, (B, N, 3) point cloud normals
                - joints: jt.Var, (B, J, 3) joint positions
                - skin: jt.Var, (B, J, J) skinning weights
        """
        
        if self.preload_data and hasattr(self, 'preloaded_data') and self.preloaded_data is not None:
            # Get from preloaded data (already pre-converted)
            precomputed_data = self.preloaded_data[index]
            if precomputed_data is None:
                # If preloading failed, fallback to disk loading
                return self._load_from_disk(index)
            
            # Apply pose augmentation if needed
            if (self.use_pose_augmentation and 
                random.random() < self.pose_augmentation_ratio and
                precomputed_data.get('original_asset') is not None):
                # Use preloaded original asset for pose augmentation
                import copy
                asset = copy.deepcopy(precomputed_data['original_asset'])
                self._apply_pose_augmentation(asset)
                
                # Re-sample and convert
                sampled_asset = asset.sample(sampler=self._sampler)
                vertices = jt.array(sampled_asset.vertices, dtype=jt.float32)
                normals = jt.array(sampled_asset.normals, dtype=jt.float32)
                
                # Process joints and skin
                if sampled_asset.joints is not None:
                    joints = jt.array(sampled_asset.joints, dtype=jt.float32)
                    if joints.shape[0] < 52:
                        padded_joints = jt.zeros((52, 3), dtype=jt.float32)
                        padded_joints[:joints.shape[0]] = joints
                        joints = padded_joints
                else:
                    joints = precomputed_data['joints']
                
                if sampled_asset.skin is not None:
                    skin = jt.array(sampled_asset.skin, dtype=jt.float32)
                    if skin.shape[1] < 52:
                        padded_skin = jt.zeros((skin.shape[0], 52), dtype=jt.float32)
                        padded_skin[:, :skin.shape[1]] = skin
                        skin = padded_skin
                else:
                    skin = precomputed_data['skin']
                
                res = {
                    'vertices': vertices,
                    'normals': normals,
                    'joints': joints,
                    'skin': skin,
                    'cls': precomputed_data['cls'],
                    'id': precomputed_data['id'],
                }
                if self.return_origin_vertices:
                    res['origin_vertices'] = jt.array(asset.vertices.copy(), dtype=jt.float32)
                return res
            elif (self.use_pose_augmentation and 
                  random.random() < self.pose_augmentation_ratio):
                # Fallback to disk loading if no preloaded original asset
                return self._load_from_disk_with_augmentation(index)
            
            # Return pre-converted data directly (no need for deep copy, as jittor tensors are immutable or use copy-on-write)
            res = {
                'vertices': precomputed_data['vertices'],
                'normals': precomputed_data['normals'],
                'cls': precomputed_data['cls'],
                'id': precomputed_data['id'],
            }
            if precomputed_data['joints'] is not None:
                res['joints'] = precomputed_data['joints']
            if precomputed_data['skin'] is not None:
                res['skin'] = precomputed_data['skin']
            if self.return_origin_vertices and precomputed_data['origin_vertices'] is not None:
                res['origin_vertices'] = precomputed_data['origin_vertices']
            return res
        else:
            # Load from disk (maintain original logic)
            return self._load_from_disk(index)
    
    def _load_from_disk(self, index):
        """Original logic for loading data from disk"""
        path = self.paths[index]
        try:
            asset = Asset.load(os.path.join(self.data_root, path))
        except Exception as e:
            print(f"Error loading asset {path}: {e}")
            raise e
        
        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy(), dtype=jt.float32)
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices, dtype=jt.float32)
        normals     = jt.array(sampled_asset.normals, dtype=jt.float32)

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints, dtype=jt.float32)
            # Ensure joint count is 52
            if joints.shape[0] < 52:
                padded_joints = jt.zeros((52, 3), dtype=jt.float32)
                padded_joints[:joints.shape[0]] = joints
                joints = padded_joints
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin, dtype=jt.float32)
            # Ensure skin weight dimension is 52
            if skin.shape[1] < 52:
                padded_skin = jt.zeros((skin.shape[0], 52), dtype=jt.float32)
                padded_skin[:, :skin.shape[1]] = skin
                skin = padded_skin
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
        return res
    
    def _load_from_disk_with_augmentation(self, index):
        """Disk loading with pose augmentation"""
        path = self.paths[index]
        try:
            asset = Asset.load(os.path.join(self.data_root, path))
        except Exception as e:
            print(f"Error loading asset {path}: {e}")
            raise e
        
        # Apply pose augmentation
        if (self.use_pose_augmentation and 
            random.random() < self.pose_augmentation_ratio):
            self._apply_pose_augmentation(asset)
        
        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy(), dtype=jt.float32)
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices, dtype=jt.float32)
        normals     = jt.array(sampled_asset.normals, dtype=jt.float32)

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints, dtype=jt.float32)
            if joints.shape[0] < 52:
                padded_joints = jt.zeros((52, 3), dtype=jt.float32)
                padded_joints[:joints.shape[0]] = joints
                joints = padded_joints
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin, dtype=jt.float32)
            if skin.shape[1] < 52:
                padded_skin = jt.zeros((skin.shape[0], 52), dtype=jt.float32)
                padded_skin[:, :skin.shape[1]] = skin
                skin = padded_skin
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)

# Example usage of the dataset
def get_dataloader(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    use_pose_augmentation: bool = True,
    pose_augmentation_ratio: float = 0.8,
    preload_data: bool = True,
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        use_pose_augmentation (bool): Whether to use pose augmentation
        pose_augmentation_ratio (float): Ratio of samples to apply pose augmentation
        preload_data (bool): Whether to preload all data to memory for faster training
        
    Returns:
        dataset (RigDataset): The dataset
    """
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        use_pose_augmentation=use_pose_augmentation,
        pose_augmentation_ratio=pose_augmentation_ratio,
        preload_data=preload_data,
    )
    
    return dataset 