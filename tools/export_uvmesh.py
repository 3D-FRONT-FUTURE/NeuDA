# Usage: Blender --background --python export_uvmesh.py {in_mesh_fpath} {out_mesh_fpath}

import os
import bpy
import sys


def export_uvmesh(in_mesh_fpath, out_mesh_fpath, use_filter=True):
    assert out_mesh_fpath.endswith(".obj"), f"must use .obj format: {out_mesh_fpath}"

    # remove default objects
    bpy.data.objects["Camera"].select_set(True)
    bpy.data.objects["Cube"].select_set(True)
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    # import mesh
    mesh_fname = os.path.basename(in_mesh_fpath)[:-4]
    if in_mesh_fpath.endswith(".obj"):
        bpy.ops.import_scene.obj(
            filepath=in_mesh_fpath,
            use_edges=True,
            use_smooth_groups=True,
            use_split_objects=True,
            use_split_groups=True,
            use_groups_as_vgroups=False,
            use_image_search=True,
            split_mode="ON",
            global_clamp_size=0,
            axis_forward="-Z",
            axis_up="Y",
        )
    else:
        bpy.ops.import_mesh.ply(filepath=in_mesh_fpath)

    obj = bpy.data.objects[mesh_fname]

    # unwrap
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    # vertice color --> UV + texture map
    # Create a new material
    # Get material
    mat = bpy.data.materials.get("Material")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material")
    # Assign it to object
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)
    obj.data.materials[0].use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    vertex_node = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
    mat.node_tree.links.new(vertex_node.outputs["Color"], bsdf.inputs["Base Color"])

    # create a new image
    bpy.ops.image.new(name="texture", width=1024, height=1024, color=(0.0, 0.0, 0.0, 0.0), alpha=True, generated_type="BLANK", float=False, use_stereo_3d=False)
    newNode = mat.node_tree.nodes.new("ShaderNodeTexImage")
    newNode.image = bpy.data.images["texture"]

    # Switch to Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # create new UV map
    bpy.ops.mesh.uv_texture_add()
    obj.data.uv_layers.active_index = 1
    obj.data.uv_layers[-1].name = "baked"
    
    # Starting the bake
    print("Baking...")
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_selected_to_active = False, margin = 3, use_clear = True)
    print("finished")

    #Saving the image in the desired folder
    print("saving texture image")
    bpy.data.images['texture'].filepath_raw = os.path.join(os.path.dirname(out_mesh_fpath), "texture.png")
    bpy.data.images['texture'].file_format = 'PNG'
    bpy.data.images['texture'].save()
    print("image saved")


    # switching to the new UV Map
    bpy.context.view_layer.objects.active = obj
    obj.data.uv_layers["baked"].active_render = True
    obj.data.uv_layers.active_index=0
    bpy.ops.mesh.uv_texture_remove()


    # Removing all the previous materials
    for x in obj.material_slots: #For all of the materials in the selected object:
        bpy.context.object.active_material_index = 0 #select the top material
        bpy.ops.object.material_slot_remove() #delete it


    # Create a new material
    bpy.data.materials.new("baked_tex_mat")
    mat=bpy.data.materials['baked_tex_mat']
    mat.use_nodes=True

    # Create a new extra material, apparently Meshlab needs at least 2 materials to show the textures correctly
    bpy.data.materials.new("mat2")
    mat2=bpy.data.materials['mat2']

    node_tree = bpy.data.materials['baked_tex_mat'].node_tree
    node = node_tree.nodes.new("ShaderNodeTexImage")
    node.select = True
    node_tree.nodes.active = node

    node.image=bpy.data.images['texture']

    node_2=node_tree.nodes['Principled BSDF']
    node_tree.links.new(node.outputs["Color"], node_2.inputs["Base Color"])

    # Assign the material to the object
    obj.data.materials.append(mat)
    obj.data.materials.append(mat2)

    # export mesh
    if in_mesh_fpath.endswith(".obj"):
        bpy.ops.export_scene.obj(
            filepath=out_mesh_fpath,
            axis_forward="-Y",
            axis_up="-Z",
            use_selection=True,
            use_normals=True,
            use_uvs=True,
            use_materials=True, 
            use_triangles=True,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=out_mesh_fpath,
            axis_forward="Y", 
            axis_up="Z", 
            use_selection=True,
            use_normals=True,
            use_uvs=True,
            use_materials=True, 
            use_triangles=True,
        )


if __name__ == '__main__':
    print(sys.argv)
    export_uvmesh(sys.argv[-2], sys.argv[-1])
