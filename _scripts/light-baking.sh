SCENE_COLLECTION=Blender
DATASET_IS_SYNTHETIC=t

# [kitchen, livingroom]
SCENE_NAME=kitchen

NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}-relighted

EAG_WITH_OPTIMIZED_ALBEDOS_PLY_PATH=_output/${SCENE_COLLECTION}-${SCENE_NAME}_1-diffuse/iter400-plys/optimized-2d-gaussians_iter400.ply

if [ ${SCENE_NAME} = kitchen ]; then
    I_SCENE_EDITING_SCENARIO=100
elif [ ${SCENE_NAME} = livingroom ] ; then
    I_SCENE_EDITING_SCENARIO=110
else
    echo "unsupported SCENE_NAME"
fi


# First render all path-traced images in trainset.
python scene-editing-and-rendering.py \
  --NVS_DATASET_PATH ${NVS_DATASET} --EAG_PLY_PATH ${EAG_WITH_OPTIMIZED_ALBEDOS_PLY_PATH} \
  --I_SCENE_EDITING_SCENARIO ${I_SCENE_EDITING_SCENARIO} \
  --OUTPUT_FOLDER_SUFFIX ${SCENE_COLLECTION}-${SCENE_NAME}_renders_editing-scenario-${I_SCENE_EDITING_SCENARIO}_trainset \
  --RENDER_FOR_LIGHT_BAKING t



if [ ${SCENE_NAME} = kitchen ]; then
    EDITED_EAG_PLY_PATH=_output/Blender-kitchen_renders_editing-scenario-${I_SCENE_EDITING_SCENARIO}_trainset/edited.ply
    LIGHT_BAKING_TRAINSET_PATH_TRACED_FOLDER=_output/Blender-kitchen_renders_editing-scenario-${I_SCENE_EDITING_SCENARIO}_trainset/2-pathtracing-spp1024
elif [ ${SCENE_NAME} = livingroom ] ; then
    EDITED_EAG_PLY_PATH=_output/Blender-livingroom_renders_editing-scenario-${I_SCENE_EDITING_SCENARIO}_trainset/edited.ply
    LIGHT_BAKING_TRAINSET_PATH_TRACED_FOLDER=_output/Blender-livingroom_renders_editing-scenario-${I_SCENE_EDITING_SCENARIO}_trainset/2-pathtracing-spp1024/
else
    echo "unsupported SCENE_NAME"
fi



LEARNING_RATE_POSITION=0.0
LEARNING_RATE_SCALE=0.0
LEARNING_RATE_QUATERNION=0.0
LEARNING_RATE_OPACITY=0.0
# LEARNING_RATE_RADIANCE=0.0
LEARNING_RATE_EMISSIVE=0.0



# Then bake the radiance into radiant 2D Gaussians.
python light-baking.py \
  --NVS_DATASET_PATH ${NVS_DATASET} --DATASET_IS_SYNTHETIC ${DATASET_IS_SYNTHETIC} \
  --EAG_PLY_PATH ${EDITED_EAG_PLY_PATH} \
  --OUTPUT_FOLDER_SUFFIX ${SCENE_COLLECTION}-${SCENE_NAME}_light-baking_editing-scenario-${I_SCENE_EDITING_SCENARIO} \
  --LEARNING_RATE_POSITION ${LEARNING_RATE_POSITION} --LEARNING_RATE_SCALE ${LEARNING_RATE_SCALE} --LEARNING_RATE_QUATERNION ${LEARNING_RATE_QUATERNION} --LEARNING_RATE_OPACITY ${LEARNING_RATE_OPACITY} --LEARNING_RATE_EMISSIVE ${LEARNING_RATE_EMISSIVE} \
  --LIGHT_BAKING_TRAINSET_PATH_TRACED_FOLDER ${LIGHT_BAKING_TRAINSET_PATH_TRACED_FOLDER} \
  --ITERATION 3000
