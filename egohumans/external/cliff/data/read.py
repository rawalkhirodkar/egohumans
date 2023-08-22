import numpy as np
import os
import trimesh

file = '/home/rawalk/Desktop/ego/cliff/data/test_samples/temp/temp_cliff_hr48.npz'
output_dir='/home/rawalk/Desktop/ego/cliff/data/test_samples/temp'

with np.load(file) as data:
    all_vertices = data['pred_verts']
    faces = data['faces']
    translation = data['global_t']
    imgnames = data['imgname']

mesh_output_dir = os.path.join(output_dir, 'meshes')
if not os.path.exists(mesh_output_dir):
    os.makedirs(mesh_output_dir)

for idx in range(len(all_vertices)):
    imgname = imgnames[0]
    imgname = imgname.split('/')[-1]
    imgname = imgname.replace('.jpg', '').replace('.png', '')

    vertices = all_vertices[idx]
    trans = translation[idx]
    vertices = vertices - trans ## for cliff render

    # vertices = vertices - vertices.mean(axis=0)

    mesh = trimesh.Trimesh(vertices, faces)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]) ## flip x axis
    mesh.apply_transform(rot)

    save_id = 'mesh_{}_{}'.format(imgname, idx)
    mesh.export(os.path.join(mesh_output_dir, '{}.obj'.format(save_id)))

    np.save(os.path.join(mesh_output_dir,'{}.npy'.format(save_id)), trans)