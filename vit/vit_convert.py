#wd14-taggerを直下にダウンロードします。

from huggingface_hub import hf_hub_download
import os
from vit import ViT
from tensorflow.keras.models import load_model

DEFAULT_WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-vit-tagger-v2'
TAGGER_DIR = 'wd-v1-4-vit-tagger-v2'
FILES = ["keras_metadata.pb", "saved_model.pb","selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]

def download(path):
    model_dir = os.path.join(path, TAGGER_DIR)
    if not os.path.exists(model_dir):
        print(f"downloading wd14 tagger model from hf_hub. id: {DEFAULT_WD14_TAGGER_REPO}")
        for file in FILES:
            hf_hub_download(DEFAULT_WD14_TAGGER_REPO, file, cache_dir=model_dir, force_download=True, force_filename=file)
        for file in SUB_DIR_FILES:
            hf_hub_download(DEFAULT_WD14_TAGGER_REPO, file, subfolder=SUB_DIR, cache_dir=os.path.join(
                model_dir, SUB_DIR), force_download=True, force_filename=file)
    else:
        print("using existing wd14 tagger model")

download("./")

model = load_model("wd-v1-4-vit-tagger-v2")

vit = ViT(3,448,9083)
state_dict = vit.state_dict()

new_dic = {}
def tf_weight(str,i):
    return model.get_layer(str).get_weights()[i]

def assert_unmatch_key(key):
    assert new_dic[key].shape == state_dict[key].shape, (key, new_dic[key].shape, state_dict[key].shape)

def convert(torch_key, tf_key, tf_index, dence=False, conv=False):
    global new_dic, state_dict
    new_dic[torch_key] = torch.tensor(tf_weight(tf_key,tf_index))
    if dence:
        new_dic[torch_key] = new_dic[torch_key].transpose(1,0)
    if conv:
        new_dic[torch_key] = new_dic[torch_key].permute(3,2,0,1)
    assert_unmatch_key(torch_key)

#conv_in
convert("conv.weight","root_conv2d_01",0,conv=True)
convert("conv.bias","root_conv2d_01",1)

#pos_embed
convert("pos_embed.pos_embed","pos_embed",0)
#layernorm
for block in range(12):
    for j in [1,2]:
        convert(f"blocks.{block}.norm{j}.weight", f"block{block}_norm_0{j}",0)
        convert(f"blocks.{block}.norm{j}.bias", f"block{block}_norm_0{j}",1)

convert(f"norm.weight", f"predictions_norm",0)
convert(f"norm.bias", f"predictions_norm",1)
#skip
for block in range(12):
    for j in [1,2]:
        if block*2 + j - 1 == 0:
            convert(f"blocks.{block}.skip{j}.skip",f"skip_init_channelwise",0)
        else:
            convert(f"blocks.{block}.skip{j}.skip",f"skip_init_channelwise_{block*2 + j - 1}",0)
#mlp
for block in range(12):
    for j in [1,2]:
        convert(f"blocks.{block}.mlp.fc{j}.weight",f"block{block}_cm_dense_0{j}",0,True)
        convert(f"blocks.{block}.mlp.fc{j}.bias",f"block{block}_cm_dense_0{j}",1)
#multi head attention
for block in range(12):
    #weight
    tf_key = f"multi_head_attention_{block}" if block > 0 else f"multi_head_attention"
    query = torch.tensor(tf_weight(tf_key, 0)).reshape(768,-1).transpose(1,0)
    key = torch.tensor(tf_weight(tf_key, 2)).reshape(768,-1).transpose(1,0)
    value = torch.tensor(tf_weight(tf_key, 4)).reshape(768,-1).transpose(1,0)
    new_dic[f"blocks.{block}.attn.in_proj_weight"] = torch.cat([query,key,value], dim = 0)
    assert_unmatch_key(f"blocks.{block}.attn.in_proj_weight")

    #bias
    query = torch.tensor(tf_weight(tf_key, 1)).reshape(-1)
    key = torch.tensor(tf_weight(tf_key, 3)).reshape(-1)
    value = torch.tensor(tf_weight(tf_key, 5)).reshape(-1)
    new_dic[f"blocks.{block}.attn.in_proj_bias"] = torch.cat([query,key,value], dim = 0)
    assert_unmatch_key(f"blocks.{block}.attn.in_proj_bias")

    #out
    weight = torch.tensor(tf_weight(tf_key, 6)).reshape(768,-1).transpose(1,0)
    bias = torch.tensor(tf_weight(tf_key, 7)).reshape(-1)
    new_dic[f"blocks.{block}.attn.out_proj.weight"] = weight
    new_dic[f"blocks.{block}.attn.out_proj.bias"] = bias
    assert_unmatch_key(f"blocks.{block}.attn.out_proj.weight")
    assert_unmatch_key(f"blocks.{block}.attn.out_proj.bias")
#last fully connected
convert("fc.weight","predictions_dense",0,True)
convert("fc.bias","predictions_dense",1)
vit.load_state_dict(new_dic)
torch.save(vit.state_dict(),"wd-v1-4-vit-tagger-v2.ckpt")
