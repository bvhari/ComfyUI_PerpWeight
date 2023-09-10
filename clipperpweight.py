import torch


class CLIPTextEncodePerpWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "clip": ("CLIP", ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        empty_tokens = clip.tokenize("")
        empty_cond = clip.encode_from_tokens(empty_tokens, return_pooled=False)
        tokens = clip.tokenize(text)
        unweighted_tokens = [[(t, 1.0) for t,_ in x] for x in tokens]
        unweighted_cond, unweighted_pooled = clip.encode_from_tokens(unweighted_tokens, return_pooled=True)

        cond = torch.clone(unweighted_cond)
        for i in range(unweighted_cond.shape[0]):
            for j in range(unweighted_cond.shape[1]):
                weight = tokens[i][j][1]
                if weight != 1.0:
                    token_vector = unweighted_cond[i][j]
                    zero_vector = empty_cond[i][j]
                    perp = ((torch.mul(zero_vector, token_vector).sum())/(torch.norm(token_vector)**2)) * token_vector
                    cond[i][j] = token_vector + (weight * perp)
        
        return ([[cond, {"pooled_output": unweighted_pooled}]], )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeMultiLayer": CLIPTextEncodePerpWeight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeMultiLayer": "CLIP Text Encode (Perp-Weight)",
}
