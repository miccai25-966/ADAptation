import torch
import torch.nn as nn
import open_clip
import os
os.environ['HF_HUB_DISABLE_CACHE'] = '1'

class BiomedCLIPTextEncoder(nn.Module):    #一个强行逆天改命，将原本的context_length=256，缩成了77
    def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', context_length=77, device='cuda',): # context_length修改为128
        super().__init__()
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            'hf-hub:' + model_name,
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)
        self.context_length = context_length
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def encode_text(self, text):
        text_tokens = self.tokenizer(text,  context_length=self.context_length).to(self.device)
        print("4433344", text_tokens.shape)  # 输出的shape会根据context_length变化

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            print("2112", text_features.shape)

            # # 添加线性层
            # self.projection = nn.Linear(512, 768).to(self.device) # 初始化线性层，并将其移动到设备上
            # text_features = self.projection(text_features) # 将特征向量映射到 768 维
            
        result = torch.einsum('bi,bj->bij', text_tokens, text_features)
        print("output", result.shape)
        return result

    def forward(self, text):
        return self.encode_text(text)

# 测试代码
if __name__ == "__main__":
    texts = ["This is a test sentence.", "Another test sentence here.", "A longer sentence to test truncation."]
    encoder = BiomedCLIPTextEncoder(context_length=77) # 初始化时设置context_length
    embeddings = encoder(texts)
    print(f"Embeddings shape: {embeddings.shape}")






# import torch
# import torch.nn as nn
# import open_clip

# class BiomedCLIPTextEncoder(nn.Module):
#     def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', context_length=77, device='cuda'):
#         super().__init__()

#         # 使用模型名称或 Hugging Face Hub ID 加载模型
#         self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
#             model_name  # 直接使用模型名称
#         )
#         self.tokenizer = open_clip.get_tokenizer(model_name)  # 直接使用模型名称

#         self.context_length = context_length
#         self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
#         self.model.to(self.device)

#     def encode_text(self, text):
#         text_tokens = self.tokenizer(text,  context_length=self.context_length).to(self.device)
#         print("4433344", text_tokens.shape)  # 输出的shape会根据context_length变化

#         with torch.no_grad():
#             text_features = self.model.encode_text(text_tokens)
#             print("2112", text_features.shape)

#             # # 添加线性层
#             # self.projection = nn.Linear(512, 768).to(self.device) # 初始化线性层，并将其移动到设备上
#             # text_features = self.projection(text_features) # 将特征向量映射到 768 维
            
#         result = torch.einsum('bi,bj->bij', text_tokens, text_features)
#         print("output", result.shape)
#         return result

#     def forward(self, text):
#         return self.encode_text(text).cpu()

# # 测试代码
# if __name__ == "__main__":
#     texts = ["This is a test sentence.", "Another test sentence here.", "A longer sentence to test truncation."]
#     encoder = BiomedCLIPTextEncoder(context_length=77) # 初始化时设置context_length
#     embeddings = encoder(texts)
#     print(f"Embeddings shape: {embeddings.shape}")
        


# import torch
# import torch.nn as nn
# import open_clip
# from huggingface_hub import snapshot_download
# import os

# class BiomedCLIPTextEncoder(nn.Module):
#     def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', context_length=77, device='cuda', local_model_path="A"): # 添加local_model_path参数
#         super().__init__()

#         # 指定本地模型路径
#         self.local_model_path = os.path.abspath(local_model_path) # 获取绝对路径
#         if not os.path.exists(self.local_model_path):
#             os.makedirs(self.local_model_path) # 如果文件夹不存在，则创建

#         # 使用snapshot_download下载模型到本地
#         snapshot_download(repo_id=model_name, local_dir=self.local_model_path, local_dir_use_symlinks=False)

#         self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
#             self.local_model_path # 使用本地模型路径
#         )
#         self.tokenizer = open_clip.get_tokenizer(self.local_model_path) # 使用本地模型路径
#         self.context_length = context_length
#         self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
#         self.model.to(self.device)

#     def encode_text(self, text):
#         text_tokens = self.tokenizer(text,  context_length=self.context_length).to(self.device)
#         print("4433344", text_tokens.shape)  # 输出的shape会根据context_length变化

#         with torch.no_grad():
#             text_features = self.model.encode_text(text_tokens)
#             print("2112", text_features.shape)

#             # # 添加线性层
#             # self.projection = nn.Linear(512, 768).to(self.device) # 初始化线性层，并将其移动到设备上
#             # text_features = self.projection(text_features) # 将特征向量映射到 768 维
            
#         result = torch.einsum('bi,bj->bij', text_tokens, text_features)
#         print("output", result.shape)
#         return result

#     def forward(self, text):
#         return self.encode_text(text).cpu()


# # 测试代码
# if __name__ == "__main__":
#     local_path = "/home/data/duanyaofei/ControlNet-main-1022/models/Bio" # 指定本地文件夹
#     texts = ["This is a test sentence.", "Another test sentence here.", "A longer sentence to test truncation."]
#     encoder = BiomedCLIPTextEncoder(context_length=77, local_model_path=local_path) # 传递local_model_path参数
#     embeddings = encoder(texts)
#     print(f"Embeddings shape: {embeddings.shape}")
