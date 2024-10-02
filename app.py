import argparse
from pathlib import Path
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
from PIL import Image
import os
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List

CLIP_PATH = "google/siglip-so400m-patch14-384"
TITLE = "<h1><center>JoyCaption Alpha Two (2024-09-26a)</center></h1>"
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

EXTRA_OPTIONS_LIST = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]

CAPTION_LENGTH_CHOICES = (
    ["any", "very short", "short", "medium-length", "long", "very long"]
    + [str(i) for i in range(20, 261, 10)]
)

HF_TOKEN = os.environ.get("HF_TOKEN", None)


class ImageAdapter(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        ln1: bool,
        pos_emb: bool,
        num_image_tokens: int,
        deep_extract: bool,
    ):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = (
            None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        )

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(
            mean=0.0, std=0.02
        )  # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert (
                x.shape[-1] == vision_outputs[-2].shape[-1] * 5
            ), f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (
            x.shape[0],
            2,
            x.shape[2],
        ), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


# Determine the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

if device.type == "cuda":
    autocast = torch.cuda.amp.autocast
else:
    autocast = contextlib.nullcontext  # No autocasting on CPU


def load_models(CHECKPOINT_PATH):
    # Load CLIP
    print("Loading CLIP")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model

    assert (
        CHECKPOINT_PATH / "clip_model.pt"
    ).exists(), f"clip_model.pt not found in {CHECKPOINT_PATH}"
    print("Loading VLM's custom vision model")
    checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location="cpu")
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to(device)

    # Tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH / "text_model", use_fast=True
    )
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

    # LLM
    print("Loading LLM")
    print("Loading VLM's custom text model")
    text_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH / "text_model", torch_dtype=torch_dtype
    )
    text_model.eval()
    text_model.to(device)

    # Image Adapter
    print("Loading image adapter")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False
    )
    image_adapter.load_state_dict(
        torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu")
    )
    image_adapter.eval()
    image_adapter.to(device)

    return clip_processor, clip_model, tokenizer, text_model, image_adapter


@torch.no_grad()
def stream_chat(
    input_image: Image.Image,
    caption_type: str,
    caption_length: Union[str, int],
    extra_options: List[str],
    name_input: str,
    custom_prompt: str,
    clip_model,
    tokenizer,
    text_model,
    image_adapter,
) -> tuple:
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 'any' means no length specified
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    # Build prompt
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")

    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Add extra options
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)

    # Add name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    print(f"Prompt: {prompt_str}")

    # Preprocess image
    # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
    # image = clip_processor(images=input_image, return_tensors='pt').pixel_values
    image = input_image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.to(device)

    # Embed image
    # This results in Batch x Image Tokens x Features
    with autocast():
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(device)

    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt_str,
        },
    ]

    # Format the conversation
    # The apply_chat_template method might not be available; handle accordingly
    if hasattr(tokenizer, "apply_chat_template"):
        convo_string = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
    else:
        # Simple concatenation if apply_chat_template is not available
        convo_string = (
            "<|system|>" + convo[0]["content"] + "<|end|><|user|>" + convo[1]["content"] + "<|end|>"
        )

    assert isinstance(convo_string, str)

    # Tokenize the conversation
    # prompt_str is tokenized separately so we can do the calculations below
    convo_tokens = tokenizer.encode(
        convo_string, return_tensors="pt", add_special_tokens=False, truncation=False
    )
    prompt_tokens = tokenizer.encode(
        prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False
    )
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
    prompt_tokens = prompt_tokens.squeeze(0)

    # Calculate where to inject the image
    eot_id_indices = (
        (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        .nonzero(as_tuple=True)[0]
        .tolist()
    )
    if len(eot_id_indices) != 2:
        # Fallback if <|eot_id|> tokens are not present
        eot_id_indices = [0, len(convo_tokens)]
    preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt

    # Embed the tokens
    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

    # Construct the input
    input_embeds = torch.cat(
        [
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ],
        dim=1,
    ).to(device)

    input_ids = torch.cat(
        [
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),  # Dummy tokens for the image
            convo_tokens[preamble_len:].unsqueeze(0),
        ],
        dim=1,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Debugging
    print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

    # Generate the caption
    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        suppress_tokens=None,
    )

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    return prompt_str, caption.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning")

    parser.add_argument(
        "input_image",
        type=str,
        nargs="?",
        help="Path to the input image",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="captions.txt",
        help="Path to the output file where captions will be saved",
    )
    parser.add_argument(
        "--caption_type",
        type=str,
        default="Descriptive",
        choices=CAPTION_TYPE_MAP.keys(),
        help="Type of caption to generate",
    )
    parser.add_argument(
        "--caption_length",
        type=str,
        default="long",
        choices=CAPTION_LENGTH_CHOICES,
        help="Length of the caption",
    )
    parser.add_argument(
        "--extra_options",
        type=int,
        nargs="*",
        default=[],
        help="Indices of extra options to customize the caption",
    )
    parser.add_argument(
        "--name_input",
        type=str,
        default="",
        help="Person/Character Name (if applicable)",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default="",
        help="Custom prompt (overrides other settings)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="cgrkzexw-599808",
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--list_extra_options",
        action="store_true",
        help="List available extra options",
    )

    args = parser.parse_args()

    if args.list_extra_options:
        print("Available extra options:")
        for idx, option in enumerate(EXTRA_OPTIONS_LIST):
            print(f"{idx}: {option}")
        exit(0)

    if args.input_image is None and args.input_dir is None:
        parser.error("You must specify either an input image or an input directory.")

    if args.input_image and args.input_dir:
        parser.error("Please specify either an input image or an input directory, not both.")

    # Map extra option indices to actual options
    if args.extra_options:
        extra_options = []
        for idx in args.extra_options:
            if 0 <= idx < len(EXTRA_OPTIONS_LIST):
                extra_options.append(EXTRA_OPTIONS_LIST[idx])
            else:
                print(f"Invalid extra option index: {idx}")
                exit(1)
    else:
        extra_options = []

    name_input = args.name_input
    custom_prompt = args.custom_prompt
    CHECKPOINT_PATH = Path(args.checkpoint_path)

    # Load models
    (
        clip_processor,
        clip_model,
        tokenizer,
        text_model,
        image_adapter,
    ) = load_models(CHECKPOINT_PATH)

    # Process single image
    if args.input_image:
        input_image_path = Path(args.input_image)
        if not input_image_path.is_file():
            print(f"Input image not found: {input_image_path}")
            exit(1)
        input_image = Image.open(input_image_path)

        # Call the function
        prompt_str, caption = stream_chat(
            input_image,
            args.caption_type,
            args.caption_length,
            extra_options,
            name_input,
            custom_prompt,
            clip_model,
            tokenizer,
            text_model,
            image_adapter,
        )

        # Print the outputs
        print(f"\nImage: {input_image_path}")
        print(f"Prompt used:\n{prompt_str}\n")
        print(f"Caption:\n{caption}\n")

    # Process images in a directory
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Input directory not found: {input_dir}")
            exit(1)

        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        # Collect all image files in the directory
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in directory: {input_dir}")
            exit(1)

        output_file_path = Path(args.output_file)
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            for image_path in image_files:
                print(f"\nProcessing image: {image_path}")
                input_image = Image.open(image_path)

                try:
                    prompt_str, caption = stream_chat(
                        input_image,
                        args.caption_type,
                        args.caption_length,
                        extra_options,
                        name_input,
                        custom_prompt,
                        clip_model,
                        tokenizer,
                        text_model,
                        image_adapter,
                    )
                    # Write the caption to the output file
                    output_file.write(f"Image: {image_path}\n")
                    output_file.write(f"Prompt used:\n{prompt_str}\n")
                    output_file.write(f"Caption:\n{caption}\n")
                    output_file.write("\n" + "=" * 50 + "\n")
                    print(f"Caption generated for image: {image_path}")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

        print(f"\nCaptions have been saved to {output_file_path}")

