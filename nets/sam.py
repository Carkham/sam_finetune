from segment_anything import sam_model_registry


def get_model():
    return sam_model_registry['vit_h'](checkpoint='pretrain/sam_vit_h_4b8939.pth')