#!/bin/bash

export case=$1
export conf=$2
export gpu_id=$3

if [ -n "$4" ];then
    export in_mesh_path=$4
else
    export in_mesh_path="exp/$case/neuda_wmask/meshes/00300000_tex.ply"
fi

if [ -n "$5" ];then
    export out_mesh_path=$5
else
    export out_mesh_path="exp/$case/neuda_wmask/meshes/00300000_uvtex.obj"
fi

export CUDA_VISIBLE_DEVICES=$gpu_id

echo "RECONSTRUCT CASE: $case" 
echo "CONF: $conf" 
echo "GPU_ID = $gpu_id"
echo "in_mesh_path: $in_mesh_path"
echo "out_mesh_path: $out_mesh_path"

echo "python reconstruct_mesh.py --case $case --conf confs/neuda_wmask.conf --is_continue --mode validate_texture_mesh"
python reconstruct_mesh.py --case $case --conf confs/neuda_wmask.conf --is_continue --mode validate_texture_mesh

export tmp_mesh_path="${in_mesh_path%.*}_denoise.ply"
echo "python filter_mesh_noise.py $in_mesh_path $tmp_mesh_path"
python tools/filter_mesh_noise.py ${in_mesh_path} ${tmp_mesh_path}

echo "Blender --background --python export_uv.py $tmp_mesh_path $out_mesh_path"
blender-3.4 --background --python tools/export_uvmesh.py ${tmp_mesh_path} ${out_mesh_path}

echo "python refine_texture.py --mode train --conf $conf --case $case"
python refine_texture.py --mode train --conf $conf --case $case

echo "Texture fine-tuning train & evaluation done!"
