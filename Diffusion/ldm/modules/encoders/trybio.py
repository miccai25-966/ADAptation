import torch
#from transformers import create_model_from_pretrained, get_tokenizer
#from transformers import AutoTokenizer, AutoModel
# from open_clip import create_model_and_transforms, get_tokenizer
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import open_clip

# class BiomedCLIPClassifier(nn.Module):

#     def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
#                  context_length=256, device='cuda', layer="pooled"):  # Default layer is now "pooled"
#         super().__init__()
#         self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
#             'hf-hub:' + model_name
#         )
#         self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)


#         self.context_length = context_length
#         self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
#         self.model.to(self.device)
#         # self.model.eval()
    
#     def encode_text(self, text, context_length=256):
#         text_tokens = self.tokenizer(text, context_length=context_length).to(self.device)
#         print("4433344", text_tokens.shape)  # ([3, 256])
        
#         with torch.no_grad():
#             text_features = self.model.encode_text(text_tokens).to(self.device)
#             print("2112", text_features.shape)
#             # text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(self.device)
#             # print("2112", text_features.shape)
#         return text_features
    
#     def forward(self, text, context_length=256):
       
#         return self.encode_text(text, context_length=256).cpu()
    
        # tokens = self.tokenizer(texts, context_length=context_length
        #                        ).to(self.device)
        # print("4433344", tokens.shape)
        # with torch.no_grad():
        #     outputs = self.model(**tokens)
        #     print("2112", outputs.shape)
            
        # return outputs.last_hidden_state.cpu().numpy()



class BiomedCLIPClassifier(nn.Module):

    def __init__(self, model_name='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                 context_length=256, device='cuda', layer="pooled"):  # Default layer is now "pooled"
        super().__init__()
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            'hf-hub:' + model_name
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + model_name)


        self.context_length = context_length
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        # self.model.eval()
    
    def _encode_text(self, text, context_length=256):
        text_tokens = self.tokenizer(text).to(self.device)
        print("4433344", text_tokens.shape)  # ([3, 256])
        
        
        
        # with torch.no_grad():
        #     text_features = self.model.encode_text(text_tokens).to(self.device)
        #     print("2112", text_features.shape)
        #     # text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(self.device)
        #     # print("2112", text_features.shape)
        # return text_features.last_hidden_state
        with torch.no_grad():
            text_features = self.model(text_tokens).to(self.device)
            print("2112", text_features.shape)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(self.device)
            # print("2112", text_features.shape)
        return text_features.last_hidden_state
    
    
    def forward(self, text, context_length=256):
       
        return self._encode_text(text, context_length=256).cpu()
    
    

# 测试代码
if __name__ == "__main__":
    texts = ["This is a test sentence.", "Another test sentence here.", "A longer sentence to test truncation."]
    classifier = BiomedCLIPClassifier()
    embeddings = classifier.forward(texts)

    print(f"Embeddings shape: {embeddings.shape}")
    # print(f"First embedding:\n{embeddings[0]}")

