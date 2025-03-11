# import torch
# import torch.nn as nn
# import open_clip

# class BiomedCLIPTextEncoder(nn.Module):
#     def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', context_length=256, device='cuda'):
#         super().__init__()
#         self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
#             'hf-hub:' + model_name
#         )
#         self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)
#         self.context_length = context_length
#         self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
#         self.model.to(self.device)

#     def encode_text(self, text):
#         text_tokens = self.tokenizer(text).to(self.device)
#         print("4433344", text_tokens.shape)  # ([3, 256])
        
#         with torch.no_grad():
#             text_features = self.model.encode_text(text_tokens)  # 使用encode_text方法
#             print("2112", text_features.shape)
#         return text_features

#     def forward(self, text):
#         return self.encode_text(text).cpu()

# # 测试代码
# if __name__ == "__main__":
#     texts = ["This is a test sentence.", "Another test sentence here.", "A longer sentence to test truncation."]
#     encoder = BiomedCLIPTextEncoder()
#     embeddings = encoder(texts)
#     print(f"Embeddings shape: {embeddings.shape}")



import torch
import torch.nn as nn
import open_clip

class BiomedCLIPTextEncoder(nn.Module):
    def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', context_length=256, device='cuda'):
        super().__init__()
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            'hf-hub:' + model_name
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)
        self.context_length = context_length
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def encode_text(self, text):
        text_tokens = self.tokenizer(text).to(self.device)
        print("4433344", text_tokens.shape)  # ([3, 256])
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)  # 使用encode_text方法
            print("2112", text_features.shape) # ([3, 512])
        
        result = torch.einsum('bi,bj->bij', text_tokens, text_features)  # ([3, 512])

        print("output", result.shape)
        return result

    def forward(self, text):
        return self.encode_text(text).cpu()

# 测试代码
if __name__ == "__main__":
    texts = ["This is a test sentence.", "Another test sentence here.", "A longer sentence to test truncation."]
    encoder = BiomedCLIPTextEncoder()
    embeddings = encoder(texts)
    print(f"Embeddings shape: {embeddings.shape}")





