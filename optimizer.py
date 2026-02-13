# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Load model and processor
# print("[INFO] Loading model...")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model.eval()

# # Load image
# print("[INFO] Loading image...")
# image = Image.open("img.jpg").convert("RGB")
# inputs = processor(images=image, return_tensors="pt")

# # Only export the vision encoder for now (image -> embedding)
# print("[INFO] Extracting vision encoder...")
# vision_encoder = model.vision_model

# # Dummy input
# dummy_pixel_values = inputs['pixel_values']

# # Export vision encoder
# print("[INFO] Exporting vision encoder to ONNX...")
# torch.onnx.export(
#     vision_encoder,
#     (dummy_pixel_values,),
#     "blip_vision_encoder.onnx",
#     input_names=["pixel_values"],
#     output_names=["encoder_outputs"],
#     dynamic_axes={"pixel_values": {0: "batch_size"}, "encoder_outputs": {0: "batch_size"}},
#     opset_version=13
# )

# print("Exported to blip_vision_encoder.onnx")



from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

image = Image.open("img.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to("cpu")

output = model.generate(**inputs)
caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)

print("Caption:", caption)

















import torch
from torch import nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Load image and text
image = Image.open("img.jpg").convert("RGB")
text = "a photo of"

inputs = processor(images=image, text=text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
pixel_values = inputs["pixel_values"]

# Wrap model in a class that uses only positional arguments
class WrappedBLIP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        ).logits

wrapped_model = WrappedBLIP(model)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    args=(input_ids, attention_mask, pixel_values),
    f="blip_caption.onnx",
    input_names=["input_ids", "attention_mask", "pixel_values"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "pixel_values": {0: "batch"}
    },
    opset_version=13
)

print("✅ Exported to ONNX successfully.")

