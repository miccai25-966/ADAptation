import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import open_clip
# from ldm.util import default, count_params
import torch
from urllib.request import urlopen
from PIL import Image
import os
import torch
# torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
            
    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
       # print(f"ORI Tokenized input: {tokens}, ORI shape: {tokens.shape}")
   #     #torch.Size([2, 77])
        # outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)
    
    

class BiomedCLIPTextEncoder(AbstractEncoder):    #一个强行逆天改命，将原本的context_length=256，缩成了77
    def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', context_length=77, device='cuda'): # context_length修改为128
        super().__init__()
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            'hf-hub:' + model_name
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)
        self.context_length = context_length
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def encode(self, text):
        text_tokens = self.tokenizer(text,  context_length=self.context_length).to(self.device)
        #print("499999", text_tokens.shape)  # 输出的shape会根据context_length变化

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
           # print("27777", text_features.shape)

            # 添加线性层
            self.projection = nn.Linear(512, 768).to(self.device) # 初始化线性层，并将其移动到设备上
            text_features = self.projection(text_features) # 将特征向量映射到 768 维
            
            # #标准化、归一化
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(self.device)
            # # Normalize the tensor
            # normalized_tensor = (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-6)
           
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            
        #     # 补充0
        #      # 计算需要补零的数量
        #     padding_size = 768 - text_features.shape[-1]

        #     # 创建零填充张量
        #     padding = torch.zeros(text_features.shape[:-1] + (padding_size,), device=self.device, dtype=text_features.dtype)
        #    # padding = torch.zeros(text_features.shape[:-1] + (padding_size,), dtype=text_features.dtype)
            
        #     # 将填充张量与 text_features 拼接
        #     text_features = torch.cat([text_features, padding], dim=-1).to(self.device)
        
        
        
            # print("27777", text_features.shape)
            # print("27777", text_features)
        result = torch.einsum('bi,bj->bij', text_tokens, text_features).to(self.device)
        #print("output", result.shape)  #output torch.Size([2, 77, 768])
        return result

    def forward(self, text):
        return self.encode(text)
    
    def freeze(self):
        self.model = self.model.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False



class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


#-------------------------------------------------------------------------------------
class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',  
                 device="cuda", context_length=77, freeze=True, layer="last"): # context_length修改为128
        super().__init__()
        assert layer in self.LAYERS
        self.device = device
        model, _, _  = open_clip.create_model_and_transforms(
            'hf-hub:' + model_name
        )

        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)

        del model.visual
        self.model = model

        self.max_length = context_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        
        tokens = self.tokenizer(text, context_length=self.max_length).to(self.device)
        # tokens = open_clip.tokenize(text)
       # print(f"Tokenized input: {tokens}, shape: {tokens.shape}")    #torch.Size([2, 77])
        
        z = self.model.encode_text(tokens).to(self.device)
        self.projection = nn.Linear(512, 768).to(self.device)# 初始化线性层，并将其移动到设备上
        z = self.projection(z).to(self.device)
        
        result = z.unsqueeze(1).repeat(1, 77, 1).to(self.device)
        
        # result = torch.einsum('bi,bj->bij', tokens, z)
        return result

    # def encode_with_transformer(self, text):
    #     x = self.model.encode_text(text)
    #     x = x + self.model.positional_embedding
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.model.ln_final(x)
    #     return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
      
    
#----------------------------------------------------------------
 
# def main():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     texts = ["Ultrasound of breast tumor", "Ultrasound of breast tumor"]

#     cao = FrozenOpenCLIPEmbedder()
#     cao = cao.cpu()
#    # print(dir(cao))
    
#     cao.freeze()
#     caocao = cao(texts)    
#     print(caocao.shape)  #torch.Size([2, 512])
#     print(caocao) 
#     print('============================================')


#  #   encoder = BiomedCLIPTextEncoder(context_length=77) # 初始化时设置context_length
#     encoder_ori = FrozenCLIPEmbedder()


#   #  encoder.freeze()
#     encoder_ori.freeze()
    

#   #  embeddings = encoder(texts)
#     embeddings_ori = encoder_ori(texts)
#   #  print(f"Embeddings shape: {embeddings}")
#     print(f"Embeddings shape: {embeddings_ori.shape}")
#     print(f"Embeddings shape: {embeddings_ori}")
    
#     print('============================================')
#     # print(f"Embeddings shape: {embeddings}")
#     # print('-----------------------------------------')
#     # print(f"Embeddings shape: {embeddings_ori}")
#     print('============================================')

        
# if __name__ == "__main__":
#     main()