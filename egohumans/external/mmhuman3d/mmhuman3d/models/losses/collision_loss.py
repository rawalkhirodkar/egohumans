import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pyvista as pv
import numpy as np
import sdf.csrc as _C
import matplotlib.pyplot as plt
##---------------------------------------------------------------
class SDFFunction(Function):
    """
    Definition of SDF function
    """

    @staticmethod
    def forward(ctx, phi, faces, vertices):
        return _C.sdf(phi, faces, vertices)

    @staticmethod
    def backward(ctx):
        return None, None, None

##---------------------------------------------------------------
class SDF(nn.Module):

    def forward(self, faces, vertices, grid_size=32):
        phi = torch.zeros(vertices.shape[0], grid_size, grid_size, grid_size, device=vertices.device)
        phi = SDFFunction.apply(phi, faces, vertices)
        return phi

##---------------------------------------------------------------
class CollisionLoss(nn.Module):
    """MSELoss for 2D and 3D keypoints.
    """

    def __init__(self, grid_size=256, scale_factor=0.2, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.grid_size = grid_size
        self.scale_factor = scale_factor
        self.sdf = SDF()
        return
    
    def get_bounding_box3d(self, vertices):
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    def forward(self,
                vertices_dict,
                mesh_faces,
                loss_weight_override=None,
                reduction_override=None,
                debug=False):
        """Forward function of loss.
            vertices_dict (dict): dict of a trajectory of vertices of the body model.
            key is the human_name
            value is the vertices of the body model, T x 6890 x 3
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)
        
        ##---------------------------------------------------------------
        human_names = list(vertices_dict.keys())
        num_humans = len(human_names)

        loss_dict = {human_name: [] for human_name in human_names}

        for time_idx in range(vertices_dict[human_names[0]].shape[0]): 
            time_stamp = time_idx + 1 ## this corresponds to the image name

            ## get the vertices of all the humans at the current time stamp
            vertices = torch.cat([vertices_dict[human_name][time_idx].unsqueeze(0) for human_name in human_names], dim=0) ## num_humans x 6890 x 3
            vertices = vertices.contiguous()

            ## local coordinate system for each mesh using 3d bbox. We normalize the vertices in the range of [-1, 1]
            boxes = self.get_bounding_box3d(vertices.detach().clone()) ## num_humans x 2 x 3. dim 2 is min and max, dim 3 is x, y, z
            boxes_center = boxes.mean(dim=1).unsqueeze(dim=1) ## num_humans x 1 x 3
            boxes_scale = (1+self.scale_factor) * 0.5*(boxes[:,1] - boxes[:,0]).max(dim=-1)[0][:,None,None] ## num_humans x 1 x 1

            with torch.no_grad():
                vertices_centered = vertices - boxes_center
                vertices_centered_scaled = vertices_centered / boxes_scale
                assert(vertices_centered_scaled.min() >= -1)
                assert(vertices_centered_scaled.max() <= 1)
                phi = self.sdf(mesh_faces, vertices_centered_scaled, self.grid_size) ## num_humans x grid_size x grid_size x grid_size

                if debug:
                    self.visualize_sdf(phi, vertices_centered_scaled, boxes_scale, sample_idx=1)

            for i, human_name in enumerate(human_names):
                weights = torch.ones(num_humans, 1, device=vertices.device)
                weights[i, 0] = 0

                ## change the coordinates of the vertices to the local coordinate system
                vertices_local = (vertices - boxes_center[i].unsqueeze(dim=0)) / boxes_scale[i].unsqueeze(dim=0) ##  num_humans x 6890 x 3
                vertices_grid = vertices_local.view(1,-1,1,1,3)

                # Sample from the phi grid
                phi_val = nn.functional.grid_sample(phi[i][None, None], vertices_grid, align_corners=True).view(num_humans, -1)
                # ignore the phi values for the i-th shape
                cur_loss = weights * phi_val ## num_humans x 6890
                
                ## sum over all the vertices and mean over all the humans
                loss = cur_loss.sum(dim=-1).mean(dim=0) ## 1
                loss_dict[human_name].append(loss) ## sum over all the time stamps
        
        ## aggregate the loss over all the time stamps
        for human_name in human_names:
            loss_dict[human_name] = torch.stack(loss_dict[human_name]).mean() * loss_weight

        return loss_dict

    ##---------------------------------------------------------------
    ## isosurface visualization using marching cubes
    def visualize_sdf(self, phi, vertices_center_scaled=None, boxes_scale=None, sample_idx=0):
        ## if phi is cuda tensor, convert it to cpu tensor
        if phi.is_cuda:
            phi = phi.detach().cpu()

        # Convert torch tensor to numpy array
        # Select the sample idx instance in the batch
        phi_np = phi.numpy()
        phi_np = phi_np[sample_idx]

        # Create a grid
        grid = pv.UniformGrid()

        # Set the grid dimensions
        grid.dimensions = np.array(phi_np.shape)

        # Edit the spatial reference
        grid.origin = (-1, -1, -1)  # The bottom left corner of the data set
        grid.spacing = (2/(phi_np.shape[0]-1), 2/(phi_np.shape[1]-1), 2/(phi_np.shape[2]-1))  # These are the cell sizes along each axis

        # Assign the data values to the points
        grid.point_data["values"] = phi_np.reshape(-1, order="F")

        # Visualize several isosurfaces for different distances from the surface
        distances = np.linspace(0, phi_np.max(), num=100)

        contours = grid.contour(isosurfaces=distances)

        # Now plot the grid!
        plotter = pv.Plotter()
        plotter.add_mesh(contours, opacity=0.5, cmap="jet", clim=[phi_np.min(), phi_np.max()])
        plotter.show()

        # # Plot histogram
        # # Flatten the tensor to a 1D array
        # phi_np_1d = phi_np.flatten()

        # # Plot the histogram
        # plt.hist(phi_np_1d, bins=50, alpha=0.5, color='g', edgecolor='black')
        # plt.title('Distribution of SDF Values')
        # plt.xlabel('SDF Value')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.show()
        # plt.savefig('sdf_histogram.png')
        
        return
