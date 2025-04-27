import sys
import os
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from PIL import Image
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List
from pathlib import Path
import re  

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSizePolicy,
    QStatusBar,
    QProgressBar,
    QMainWindow,  
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

# --- Constants and Mappings ---
CLIP_PATH = "google/siglip-so400m-patch14-384"
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

# --- Device and Autocast Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

if device.type == "cuda":
    autocast = lambda: torch.amp.autocast(device_type='cuda', dtype=torch_dtype)
else:
    autocast = contextlib.nullcontext

# --- ImageAdapter Class ---
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
        )

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
            assert len(x.shape) == 3
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2])
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# --- load_models Function ---
def load_models(CHECKPOINT_PATH, status_callback=None):
    def update_status(msg):
        if status_callback:
            status_callback(msg)
        print(msg) # Keep console output

    update_status("Loading CLIP processor...")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    update_status("Loading CLIP vision model...")
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model

    clip_model_path = CHECKPOINT_PATH / "clip_model.pt"
    if not clip_model_path.exists():
        raise FileNotFoundError(f"clip_model.pt not found in {CHECKPOINT_PATH}")

    update_status("Loading VLM's custom vision weights...")
    checkpoint = torch.load(clip_model_path, map_location="cpu")
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    update_status(f"Moving CLIP to {device}...")
    clip_model.to(device)

    update_status("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH / "text_model", use_fast=True
    )
    if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        raise TypeError(f"Tokenizer is of type {type(tokenizer)}")

    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|end|>', '<|eot_id|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    update_status(f"Added {num_added_toks} special tokens.")

    update_status("Loading LLM with 4-bit quantization (this may take time)...")
    text_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH / "text_model",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )
    )
    text_model.eval()

    if num_added_toks > 0:
        update_status("Resizing LLM token embeddings...")
        text_model.resize_token_embeddings(len(tokenizer))

    update_status("Loading image adapter...")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False
    )
    image_adapter_path = CHECKPOINT_PATH / "image_adapter.pt"
    if not image_adapter_path.exists():
        raise FileNotFoundError(f"image_adapter.pt not found in {CHECKPOINT_PATH}")

    image_adapter.load_state_dict(
        torch.load(image_adapter_path, map_location="cpu")
    )
    image_adapter.eval()
    update_status(f"Moving image adapter to {device}...")
    image_adapter.to(device)

    update_status("Models loaded successfully.")
    return clip_processor, clip_model, tokenizer, text_model, image_adapter

# --- generate_caption Function ---
@torch.no_grad()
def generate_caption(
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

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()
    else:
        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if length is None: map_idx = 0
        elif isinstance(length, int): map_idx = 1
        elif isinstance(length, str): map_idx = 2
        else: raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]
        if len(extra_options) > 0: prompt_str += " " + " ".join(extra_options)
        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    print(f"Prompt: {prompt_str}")

    try:
        image = input_image.convert("RGB")
    except Exception as e: raise ValueError(f"Error converting image to RGB: {e}")
    if image.mode != "RGB": raise ValueError(f"Image mode after conversion is {image.mode}, expected 'RGB'.")

    image = image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    pixel_values = pixel_values.to(device)

    with autocast():
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(device)

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt_str},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    else:
        convo_string = ("<|system|>\n" + convo[0]["content"] + "\n<|end|>\n<|user|>\n" + convo[1]["content"] + "\n<|end|>\n")
    assert isinstance(convo_string, str)

    convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
    prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0)
    prompt_tokens = prompt_tokens.squeeze(0)

    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_token_id is None: raise ValueError("Tokenizer missing '<|end|>' token.")
    end_token_indices = (convo_tokens == end_token_id).nonzero(as_tuple=True)[0].tolist()
    preamble_len = end_token_indices[0] + 1 if len(end_token_indices) >= 1 else 0

    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))
    input_embeds = torch.cat([
        convo_embeds[:, :preamble_len],
        embedded_images.to(dtype=convo_embeds.dtype),
        convo_embeds[:, preamble_len:],
    ], dim=1).to(device)

    input_ids = torch.cat([
        convo_tokens[:preamble_len].unsqueeze(0),
        torch.full((1, embedded_images.shape[1]), tokenizer.pad_token_id, dtype=torch.long, device=device),
        convo_tokens[preamble_len:].unsqueeze(0),
    ], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

    generate_ids = text_model.generate(
        input_ids=input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
        max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9,
        suppress_tokens=None, eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    )

    generate_ids = generate_ids[:, input_ids.shape[1]:]
    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    caption = caption.strip()
    caption = re.sub(r'\s+', ' ', caption)

    return prompt_str, caption

class CaptionApp(QMainWindow):
    def __init__(self):
        # ... (constructor unchanged) ...
        super().__init__()
        self.setWindowTitle("JoyCaption Alpha Two - Enhanced")
        self.setGeometry(100, 100, 1200, 850)
        self.setMinimumSize(1000, 750)

        self.clip_processor = None
        self.clip_model = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None
        self.models_loaded = False

        self.input_dir = None
        self.single_image_path = None
        self.selected_image_path = None
        self.image_files = []

        self.dark_mode = False

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.initUI() # Call initUI
        self.update_button_states()
        self.apply_theme()


    def initUI(self):
        # --- Left Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Input directory selection  
        dir_layout = QHBoxLayout()
        self.input_dir_button = QPushButton("Select Input Directory")
        self.input_dir_button.setToolTip("Select a folder containing images to process in batch.")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        dir_layout.addWidget(self.input_dir_button)
        self.input_dir_label = QLabel("No directory selected")
        self.input_dir_label.setWordWrap(True)
        dir_layout.addWidget(self.input_dir_label, 1)
        left_panel.addLayout(dir_layout)

        # Single image selection  
        single_img_layout = QHBoxLayout()
        self.single_image_button = QPushButton("Select Single Image")
        self.single_image_button.setToolTip("Select one image file to process.")
        self.single_image_button.clicked.connect(self.select_single_image)
        single_img_layout.addWidget(self.single_image_button)
        self.single_image_label = QLabel("No image selected")
        self.single_image_label.setWordWrap(True)
        single_img_layout.addWidget(self.single_image_label, 1)
        left_panel.addLayout(single_img_layout)

        # Caption Type  
        self.caption_type_combo = QComboBox()
        self.caption_type_combo.addItems(CAPTION_TYPE_MAP.keys())
        self.caption_type_combo.setCurrentText("Descriptive")
        self.caption_type_combo.setToolTip("Choose the style or purpose of the caption.")
        left_panel.addWidget(QLabel("Caption Type:"))
        left_panel.addWidget(self.caption_type_combo)

        # Caption Length  
        self.caption_length_combo = QComboBox()
        self.caption_length_combo.addItems(CAPTION_LENGTH_CHOICES)
        self.caption_length_combo.setCurrentText("long")
        self.caption_length_combo.setToolTip("Select desired caption length or word count.")
        left_panel.addWidget(QLabel("Caption Length:"))
        left_panel.addWidget(self.caption_length_combo)

        # Extra Options  
        left_panel.addWidget(QLabel("Extra Options:"))
        self.extra_options_checkboxes = []
        for option in EXTRA_OPTIONS_LIST:
            checkbox = QCheckBox(option)
            checkbox.setToolTip(option)
            self.extra_options_checkboxes.append(checkbox)
            left_panel.addWidget(checkbox)

        # Name Input  
        self.name_input_line = QLineEdit()
        self.name_input_line.setPlaceholderText("e.g., 'the main character'")
        self.name_input_line.setToolTip("If the first extra option is checked, this name will be used.")
        left_panel.addWidget(QLabel("Person/Character Name (optional):"))
        left_panel.addWidget(self.name_input_line)

        # Custom Prompt  
        self.custom_prompt_text = QTextEdit()
        self.custom_prompt_text.setPlaceholderText("Overrides Caption Type/Length/Options if used.")
        self.custom_prompt_text.setToolTip("Enter a full custom prompt here to ignore other settings.")
        self.custom_prompt_text.setFixedHeight(80)
        left_panel.addWidget(QLabel("Custom Prompt (optional):"))
        left_panel.addWidget(self.custom_prompt_text)

        # Checkpoint Path  
        ckpt_layout = QHBoxLayout()
        self.checkpoint_path_line = QLineEdit()
        self.checkpoint_path_line.setToolTip("Path to the folder containing model files (clip_model.pt, etc.).")
        ckpt_layout.addWidget(QLabel("Checkpoint Path:"))
        ckpt_layout.addWidget(self.checkpoint_path_line)
        self.browse_ckpt_button = QPushButton("...")
        self.browse_ckpt_button.setToolTip("Browse for Checkpoint Directory")
        self.browse_ckpt_button.clicked.connect(self.browse_checkpoint_path)
        self.browse_ckpt_button.setMaximumWidth(30)
        ckpt_layout.addWidget(self.browse_ckpt_button)
        left_panel.addLayout(ckpt_layout)

        # Load Models Button  
        self.load_models_button = QPushButton("Load Models")
        self.load_models_button.setToolTip("Load the AI models into memory (requires checkpoint path).")
        self.load_models_button.clicked.connect(self.load_models_action)
        left_panel.addWidget(self.load_models_button)

         

        # Run Buttons  
        self.run_button = QPushButton("Generate Captions for All Images in Directory")
        self.run_button.setToolTip("Process all loaded images from the selected directory.")
        self.run_button.clicked.connect(self.generate_captions_action)
        left_panel.addWidget(self.run_button)

        self.caption_selected_button = QPushButton("Caption Selected Image from List")
        self.caption_selected_button.setToolTip("Process the image currently highlighted in the list.")
        self.caption_selected_button.clicked.connect(self.caption_selected_image_action)
        left_panel.addWidget(self.caption_selected_button)

        self.caption_single_button = QPushButton("Caption Single Loaded Image")
        self.caption_single_button.setToolTip("Process the image selected via 'Select Single Image'.")
        self.caption_single_button.clicked.connect(self.caption_single_image_action)
        left_panel.addWidget(self.caption_single_button)

        # Theme Toggle Button  
        self.toggle_theme_button = QPushButton("Toggle Dark Mode")
        self.toggle_theme_button.setToolTip("Switch between light and dark themes.")
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        left_panel.addWidget(self.toggle_theme_button)

        left_panel.addStretch(1)

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # List widget for images  
        right_panel.addWidget(QLabel("Images in Directory:"))
        self.image_list_widget = QListWidget()
        self.image_list_widget.setIconSize(self.image_list_widget.iconSize() * 2)
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        self.image_list_widget.setToolTip("Click an image to view it and enable 'Caption Selected Image'.")
        right_panel.addWidget(self.image_list_widget, 1)

        # Label to display the selected image  
        right_panel.addWidget(QLabel("Selected Image Preview:"))
        self.selected_image_label = QLabel("No image selected")
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        self.selected_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selected_image_label.setMinimumSize(300, 300)
        self.selected_image_label.setStyleSheet("border: 1px solid gray;")
        right_panel.addWidget(self.selected_image_label, 3)

        # Generated Caption Area  
        right_panel.addWidget(QLabel("Generated/Editable Caption:"))
        self.generated_caption_text = QTextEdit()
        self.generated_caption_text.setReadOnly(False)
        self.generated_caption_text.setPlaceholderText("Generated caption will appear here. You can edit it before saving.")
        self.generated_caption_text.setToolTip("The generated caption appears here. Edit and use 'Save Edited Caption'.")
        right_panel.addWidget(self.generated_caption_text, 1)

         
         
        self.overwrite_checkbox = QCheckBox("Overwrite existing captions")
        self.overwrite_checkbox.setToolTip("If checked, automatically overwrites existing .txt files without asking.")
        self.append_checkbox = QCheckBox("Append to existing captions")
        self.append_checkbox.setToolTip("If checked, adds the new caption to the end of the existing .txt file.")

        # Layout for the save options
        save_options_layout = QHBoxLayout()
        save_options_layout.addWidget(self.overwrite_checkbox)
        save_options_layout.addWidget(self.append_checkbox)
        save_options_layout.addStretch(1)  
        right_panel.addLayout(save_options_layout)  

        
        self.append_checkbox.stateChanged.connect(
            lambda state: self.overwrite_checkbox.setEnabled(state == Qt.Unchecked)
        )
         

        # Save Edited Caption Button  
        self.save_caption_button = QPushButton("Save Edited Caption to File")
        self.save_caption_button.setToolTip("Save the text currently in the box above to the corresponding .txt file using the selected options.")
        self.save_caption_button.clicked.connect(self.save_edited_caption_action)
        right_panel.addWidget(self.save_caption_button)

        # --- Main Layout Assembly  
        self.main_layout.addLayout(left_panel, 2)
        self.main_layout.addLayout(right_panel, 5)

        # --- Status Bar and Progress Bar  
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        self.show_status("Ready.", 5000)

 

import sys
import os
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from PIL import Image
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List
from pathlib import Path
import re # Added for spacing fix

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSizePolicy,
    QStatusBar,
    QProgressBar,
    QMainWindow,
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

# --- Constants and Mappings ---
CLIP_PATH = "google/siglip-so400m-patch14-384"
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

# --- Device and Autocast Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

if device.type == "cuda":
    autocast = lambda: torch.amp.autocast(device_type='cuda', dtype=torch_dtype)
else:
    autocast = contextlib.nullcontext

# --- ImageAdapter Class ---
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
        )

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
            assert len(x.shape) == 3
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2])
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# --- load_models Function ---
def load_models(CHECKPOINT_PATH, status_callback=None):
    def update_status(msg):
        if status_callback:
            status_callback(msg)
        print(msg) # Keep console output

    update_status("Loading CLIP processor...")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    update_status("Loading CLIP vision model...")
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model

    clip_model_path = CHECKPOINT_PATH / "clip_model.pt"
    if not clip_model_path.exists():
        raise FileNotFoundError(f"clip_model.pt not found in {CHECKPOINT_PATH}")

    update_status("Loading VLM's custom vision weights...")
    checkpoint = torch.load(clip_model_path, map_location="cpu")
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    update_status(f"Moving CLIP to {device}...")
    clip_model.to(device)

    update_status("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH / "text_model", use_fast=True
    )
    if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        raise TypeError(f"Tokenizer is of type {type(tokenizer)}")

    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|end|>', '<|eot_id|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    update_status(f"Added {num_added_toks} special tokens.")

    update_status("Loading LLM with 4-bit quantization (this may take time)...")
    text_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH / "text_model",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )
    )
    text_model.eval()

    if num_added_toks > 0:
        update_status("Resizing LLM token embeddings...")
        text_model.resize_token_embeddings(len(tokenizer))

    update_status("Loading image adapter...")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False
    )
    image_adapter_path = CHECKPOINT_PATH / "image_adapter.pt"
    if not image_adapter_path.exists():
        raise FileNotFoundError(f"image_adapter.pt not found in {CHECKPOINT_PATH}")

    image_adapter.load_state_dict(
        torch.load(image_adapter_path, map_location="cpu")
    )
    image_adapter.eval()
    update_status(f"Moving image adapter to {device}...")
    image_adapter.to(device)

    update_status("Models loaded successfully.")
    return clip_processor, clip_model, tokenizer, text_model, image_adapter

# --- generate_caption Function ---
@torch.no_grad()
def generate_caption(
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

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()
    else:
        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if length is None: map_idx = 0
        elif isinstance(length, int): map_idx = 1
        elif isinstance(length, str): map_idx = 2
        else: raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]
        if len(extra_options) > 0: prompt_str += " " + " ".join(extra_options)
        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    print(f"Prompt: {prompt_str}")

    try:
        image = input_image.convert("RGB")
    except Exception as e: raise ValueError(f"Error converting image to RGB: {e}")
    if image.mode != "RGB": raise ValueError(f"Image mode after conversion is {image.mode}, expected 'RGB'.")

    image = image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    pixel_values = pixel_values.to(device)

    with autocast():
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(device)

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt_str},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    else:
        convo_string = ("<|system|>\n" + convo[0]["content"] + "\n<|end|>\n<|user|>\n" + convo[1]["content"] + "\n<|end|>\n")
    assert isinstance(convo_string, str)

    convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
    prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0)
    prompt_tokens = prompt_tokens.squeeze(0)

    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_token_id is None: raise ValueError("Tokenizer missing '<|end|>' token.")
    end_token_indices = (convo_tokens == end_token_id).nonzero(as_tuple=True)[0].tolist()
    preamble_len = end_token_indices[0] + 1 if len(end_token_indices) >= 1 else 0

    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))
    input_embeds = torch.cat([
        convo_embeds[:, :preamble_len],
        embedded_images.to(dtype=convo_embeds.dtype),
        convo_embeds[:, preamble_len:],
    ], dim=1).to(device)

    input_ids = torch.cat([
        convo_tokens[:preamble_len].unsqueeze(0),
        torch.full((1, embedded_images.shape[1]), tokenizer.pad_token_id, dtype=torch.long, device=device),
        convo_tokens[preamble_len:].unsqueeze(0),
    ], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

    generate_ids = text_model.generate(
        input_ids=input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
        max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9,
        suppress_tokens=None, eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    )

    generate_ids = generate_ids[:, input_ids.shape[1]:]
    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    caption = caption.strip()
    caption = re.sub(r'\s+', ' ', caption)

    return prompt_str, caption

# --- CaptionApp Class ---
class CaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JoyCaption Alpha Two - Enhanced")
        self.setGeometry(100, 100, 1200, 850)
        self.setMinimumSize(1000, 750)

        self.clip_processor = None
        self.clip_model = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None
        self.models_loaded = False

        self.input_dir = None
        self.single_image_path = None
        self.selected_image_path = None
        self.image_files = []

        self.dark_mode = False

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.initUI() # Call initUI
        self.update_button_states()
        self.apply_theme()

    def initUI(self):
        # --- Left Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Input directory selection
        dir_layout = QHBoxLayout()
        self.input_dir_button = QPushButton("Select Input Directory")
        self.input_dir_button.setToolTip("Select a folder containing images to process in batch.")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        dir_layout.addWidget(self.input_dir_button)
        self.input_dir_label = QLabel("No directory selected")
        self.input_dir_label.setWordWrap(True)
        dir_layout.addWidget(self.input_dir_label, 1)
        left_panel.addLayout(dir_layout)

        # Single image selection
        single_img_layout = QHBoxLayout()
        self.single_image_button = QPushButton("Select Single Image")
        self.single_image_button.setToolTip("Select one image file to process.")
        self.single_image_button.clicked.connect(self.select_single_image)
        single_img_layout.addWidget(self.single_image_button)
        self.single_image_label = QLabel("No image selected")
        self.single_image_label.setWordWrap(True)
        single_img_layout.addWidget(self.single_image_label, 1)
        left_panel.addLayout(single_img_layout)

        # Caption Type
        self.caption_type_combo = QComboBox()
        self.caption_type_combo.addItems(CAPTION_TYPE_MAP.keys())
        self.caption_type_combo.setCurrentText("Descriptive")
        self.caption_type_combo.setToolTip("Choose the style or purpose of the caption.")
        left_panel.addWidget(QLabel("Caption Type:"))
        left_panel.addWidget(self.caption_type_combo)

        # Caption Length
        self.caption_length_combo = QComboBox()
        self.caption_length_combo.addItems(CAPTION_LENGTH_CHOICES)
        self.caption_length_combo.setCurrentText("long")
        self.caption_length_combo.setToolTip("Select desired caption length or word count.")
        left_panel.addWidget(QLabel("Caption Length:"))
        left_panel.addWidget(self.caption_length_combo)

        # Extra Options
        left_panel.addWidget(QLabel("Extra Options:"))
        self.extra_options_checkboxes = []
        for option in EXTRA_OPTIONS_LIST:
            checkbox = QCheckBox(option)
            checkbox.setToolTip(option)
            self.extra_options_checkboxes.append(checkbox)
            left_panel.addWidget(checkbox)

        # Name Input
        self.name_input_line = QLineEdit()
        self.name_input_line.setPlaceholderText("e.g., 'the main character'")
        self.name_input_line.setToolTip("If the first extra option is checked, this name will be used.")
        left_panel.addWidget(QLabel("Person/Character Name (optional):"))
        left_panel.addWidget(self.name_input_line)

        # Custom Prompt
        self.custom_prompt_text = QTextEdit()
        self.custom_prompt_text.setPlaceholderText("Overrides Caption Type/Length/Options if used.")
        self.custom_prompt_text.setToolTip("Enter a full custom prompt here to ignore other settings.")
        self.custom_prompt_text.setFixedHeight(80)
        left_panel.addWidget(QLabel("Custom Prompt (optional):"))
        left_panel.addWidget(self.custom_prompt_text)

        # Checkpoint Path
        ckpt_layout = QHBoxLayout()
        self.checkpoint_path_line = QLineEdit()
        self.checkpoint_path_line.setToolTip("Path to the folder containing model files (clip_model.pt, etc.).")
        ckpt_layout.addWidget(QLabel("Checkpoint Path:"))
        ckpt_layout.addWidget(self.checkpoint_path_line)
        self.browse_ckpt_button = QPushButton("...")
        self.browse_ckpt_button.setToolTip("Browse for Checkpoint Directory")
        self.browse_ckpt_button.clicked.connect(self.browse_checkpoint_path)
        self.browse_ckpt_button.setMaximumWidth(30)
        ckpt_layout.addWidget(self.browse_ckpt_button)
        left_panel.addLayout(ckpt_layout)

        # Load Models Button
        self.load_models_button = QPushButton("Load Models")
        self.load_models_button.setToolTip("Load the AI models into memory (requires checkpoint path).")
        self.load_models_button.clicked.connect(self.load_models_action)
        left_panel.addWidget(self.load_models_button)

        # Run Buttons
        self.run_button = QPushButton("Generate Captions for All Images in Directory")
        self.run_button.setToolTip("Process all loaded images from the selected directory.")
        self.run_button.clicked.connect(self.generate_captions_action)
        left_panel.addWidget(self.run_button)

        self.caption_selected_button = QPushButton("Caption Selected Image from List")
        self.caption_selected_button.setToolTip("Process the image currently highlighted in the list.")
        self.caption_selected_button.clicked.connect(self.caption_selected_image_action)
        left_panel.addWidget(self.caption_selected_button)

        self.caption_single_button = QPushButton("Caption Single Loaded Image")
        self.caption_single_button.setToolTip("Process the image selected via 'Select Single Image'.")
        self.caption_single_button.clicked.connect(self.caption_single_image_action)
        left_panel.addWidget(self.caption_single_button)

        # Theme Toggle Button
        self.toggle_theme_button = QPushButton("Toggle Dark Mode")
        self.toggle_theme_button.setToolTip("Switch between light and dark themes.")
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        left_panel.addWidget(self.toggle_theme_button)

        left_panel.addStretch(1)

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # List widget for images
        right_panel.addWidget(QLabel("Images in Directory:"))
        self.image_list_widget = QListWidget()
        self.image_list_widget.setIconSize(self.image_list_widget.iconSize() * 2)
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        self.image_list_widget.setToolTip("Click an image to view it and enable 'Caption Selected Image'.")
        right_panel.addWidget(self.image_list_widget, 1)

        # Label to display the selected image
        right_panel.addWidget(QLabel("Selected Image Preview:"))
        self.selected_image_label = QLabel("No image selected")
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        self.selected_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selected_image_label.setMinimumSize(300, 300)
        self.selected_image_label.setStyleSheet("border: 1px solid gray;")
        right_panel.addWidget(self.selected_image_label, 3)

        # Generated Caption Area
        right_panel.addWidget(QLabel("Generated/Editable Caption:"))
        self.generated_caption_text = QTextEdit()
        self.generated_caption_text.setReadOnly(False)
        self.generated_caption_text.setPlaceholderText("Generated caption will appear here. You can edit it before saving.")
        self.generated_caption_text.setToolTip("The generated caption appears here. Edit and use 'Save Edited Caption'.")
        right_panel.addWidget(self.generated_caption_text, 1)

        # Saving Options  
        self.overwrite_checkbox = QCheckBox("Overwrite existing captions")
        self.overwrite_checkbox.setToolTip("If checked, automatically overwrites existing .txt files without asking.")
        self.append_checkbox = QCheckBox("Append to existing captions")
        self.append_checkbox.setToolTip("If checked, adds the new caption to the end of the existing .txt file.")

        save_options_layout = QHBoxLayout()
        save_options_layout.addWidget(self.overwrite_checkbox)
        save_options_layout.addWidget(self.append_checkbox)
        save_options_layout.addStretch(1)
        right_panel.addLayout(save_options_layout) # Add layout here

        self.append_checkbox.stateChanged.connect(
            lambda state: self.overwrite_checkbox.setEnabled(state == Qt.Unchecked)
        )

        # Save Edited Caption Button
        self.save_caption_button = QPushButton("Save Edited Caption to File")
        self.save_caption_button.setToolTip("Save the text currently in the box above to the corresponding .txt file using the selected options.")
        self.save_caption_button.clicked.connect(self.save_edited_caption_action)
        right_panel.addWidget(self.save_caption_button)

        # --- Main Layout Assembly ---
        self.main_layout.addLayout(left_panel, 2)
        self.main_layout.addLayout(right_panel, 5)

        # --- Status Bar and Progress Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        self.show_status("Ready.", 5000)

    def browse_checkpoint_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory")
        if directory:
            self.checkpoint_path_line.setText(directory)
            self.update_button_states()

    def show_status(self, message, timeout=0):
        self.status_bar.showMessage(message, timeout)
        QApplication.processEvents()

    def update_button_states(self):
        self.load_models_button.setEnabled(bool(self.checkpoint_path_line.text()))
        models_ready = self.models_loaded
        dir_selected = self.input_dir is not None and bool(self.image_files)
        single_img_selected = self.single_image_path is not None
        list_img_selected = self.selected_image_path is not None
        caption_present = bool(self.generated_caption_text.toPlainText().strip())

        self.run_button.setEnabled(models_ready and dir_selected)
        self.caption_selected_button.setEnabled(models_ready and list_img_selected)
        self.caption_single_button.setEnabled(models_ready and single_img_selected)
        self.save_caption_button.setEnabled(caption_present and (list_img_selected or single_img_selected))

    def apply_theme(self):
        dark_stylesheet = """
            QMainWindow, QWidget { background-color: #2E2E2E; color: #FFFFFF; font-family: Arial, sans-serif; }
            QPushButton { background-color: #3A3A3A; color: #FFFFFF; border: 1px solid #555555; padding: 5px; min-height: 20px; }
            QPushButton:hover { background-color: #555555; }
            QPushButton:disabled { background-color: #454545; color: #888888; }
            QLabel { color: #FFFFFF; }
            QLineEdit, QTextEdit, QComboBox { background-color: #3A3A3A; color: #FFFFFF; border: 1px solid #555555; padding: 4px; }
            QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled { background-color: #454545; color: #888888; }
            QListWidget { background-color: #3A3A3A; color: #FFFFFF; border: 1px solid #555555; alternate-background-color: #424242; }
            QCheckBox { color: #FFFFFF; spacing: 5px; }
            QCheckBox::indicator { width: 13px; height: 13px; }
            QStatusBar { color: #FFFFFF; } QStatusBar::item { border: none; }
            QProgressBar { border: 1px solid #555555; text-align: center; color: #FFFFFF; background-color: #3A3A3A; }
            QProgressBar::chunk { background-color: #007ADF; width: 10px; margin: 0.5px; }
            QToolTip { background-color: #464646; color: #FFFFFF; border: 1px solid #555555; padding: 4px; }
            QTextEdit { placeholderTextColor: gray; } QLineEdit { placeholderTextColor: gray; }
        """
        if self.dark_mode: self.setStyleSheet(dark_stylesheet)
        else: self.setStyleSheet("")

        placeholder_style = "QTextEdit { placeholderTextColor: gray; } QLineEdit { placeholderTextColor: gray; }"
        current_style = self.styleSheet()
        if self.dark_mode:
            if "placeholderTextColor" not in current_style: self.setStyleSheet(current_style + placeholder_style)
        else: self.setStyleSheet(current_style.replace(placeholder_style, ""))

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = Path(directory)
            self.input_dir_label.setText(str(self.input_dir))
            self.single_image_path = None; self.single_image_label.setText("No image selected")
            self.selected_image_path = None; self.selected_image_label.setText("No image selected")
            self.generated_caption_text.clear()
            self.load_images()
            self.show_status(f"Selected directory: {self.input_dir.name}", 5000)
        else:
            self.input_dir_label.setText("No directory selected"); self.input_dir = None
            self.image_list_widget.clear(); self.image_files = []
            self.show_status("Directory selection cancelled.", 3000)
        self.update_button_states()

    def select_single_image(self):
        file_filter = "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Single Image", "", file_filter)
        if file_path:
            self.single_image_path = Path(file_path)
            self.single_image_label.setText(str(self.single_image_path.name))
            self.input_dir = None; self.input_dir_label.setText("No directory selected")
            self.image_list_widget.clear(); self.image_files = []
            self.selected_image_path = None
            self.display_image(self.single_image_path)
            self.show_status(f"Selected single image: {self.single_image_path.name}", 5000)
        else:
            self.single_image_label.setText("No image selected"); self.single_image_path = None
            self.show_status("Single image selection cancelled.", 3000)
        self.update_button_states()

    def load_images(self):
        if not self.input_dir: return
        self.show_status(f"Loading images from {self.input_dir.name}...")
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]
        try:
            self.image_files = sorted([f for f in self.input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
        except Exception as e:
             QMessageBox.critical(self, "Directory Error", f"Could not read directory contents:\n{e}")
             self.show_status(f"Error reading directory {self.input_dir.name}", 5000)
             self.image_files = []; self.input_dir = None; self.input_dir_label.setText("Error reading directory")

        self.image_list_widget.clear()
        if not self.image_files:
            if self.input_dir:
                 QMessageBox.warning(self, "No Images", "No supported image files found.")
                 self.show_status("No images found in directory.", 3000)
            self.update_button_states()
            return

        thumb_size = 100
        for image_path in self.image_files:
            item = QListWidgetItem(str(image_path.name))
            try:
                pixmap = QPixmap(str(image_path))
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item.setIcon(QIcon(scaled_pixmap))
                else: print(f"Warning: QPixmap is null for {image_path.name}")
            except Exception as e: print(f"Warning: Could not create thumbnail for {image_path.name}: {e}")
            self.image_list_widget.addItem(item)

        self.show_status(f"Loaded {len(self.image_files)} images.", 5000)
        self.update_button_states()

    def display_selected_image(self, item):
        if not self.input_dir or not item: return
        try:
            image_name = item.text()
            image_path = self.input_dir / image_name
            if not image_path.exists():
                 QMessageBox.warning(self, "File Not Found", f"Image file '{image_name}' no longer exists.")
                 self.selected_image_label.setText("File not found")  
                 self.selected_image_label.setPixmap(QPixmap())      
                 self.generated_caption_text.clear()               
                 self.selected_image_path = None                     
                 return

            self.selected_image_path = image_path
            self.single_image_path = None  
            self.single_image_label.setText("No image selected") 
            self.display_image(image_path)
            caption_file_path = image_path.with_suffix('.txt')
            if caption_file_path.exists():
                try:
                    with open(caption_file_path, 'r', encoding='utf-8') as f:
                        caption_content = f.read()
                    self.generated_caption_text.setText(caption_content)
                    status_message = f"Displayed {image_name} and loaded existing caption."
                except Exception as e:
                    print(f"Warning: Could not read caption file {caption_file_path.name}: {e}")
                    # Keep caption box clear or show error placeholder
                    self.generated_caption_text.setPlaceholderText(f"Error reading caption file for {image_name}.")
                    status_message = f"Displayed {image_name}, but failed to load caption file."
            else:
                # Keep caption box clear (already done by display_image)
                self.generated_caption_text.setPlaceholderText("Generate or edit caption here.")
                status_message = f"Displayed {image_name}. No existing caption found."
            self.show_status(f"Selected {image_name} from list.", 4000)
        except Exception as e:
            self.selected_image_label.setText("Error loading preview")
            self.selected_image_path = None
            QMessageBox.warning(self, "Preview Error", f"Could not load preview for {item.text()}: {e}")
            self.show_status(f"Error loading preview for {item.text()}", 4000)
        self.update_button_states()

    def display_image(self, image_path):
        try:
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                self.scale_and_set_pixmap(pixmap)
                self.generated_caption_text.clear()
            else:
                self.selected_image_label.setText(f"Cannot display image:\n{image_path.name}")
                self.selected_image_label.setPixmap(QPixmap())
        except Exception as e:
            self.selected_image_label.setText(f"Error loading preview:\n{image_path.name}")
            self.selected_image_label.setPixmap(QPixmap())
            print(f"Error displaying image {image_path}: {e}")
            self.show_status(f"Error displaying image {image_path.name}", 4000)
        self.update_button_states()

    def scale_and_set_pixmap(self, pixmap):
        if not pixmap or pixmap.isNull():
             self.selected_image_label.clear()
             self.selected_image_label.setText("No image selected")
             return
        label_size = self.selected_image_label.contentsRect().size()
        scaled_pixmap = pixmap.scaled(label_size * self.devicePixelRatioF(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.selected_image_label.setPixmap(scaled_pixmap)

    def load_models_action(self):
        checkpoint_path_str = self.checkpoint_path_line.text()
        if not checkpoint_path_str: QMessageBox.warning(self, "Checkpoint Error", "Please specify the checkpoint path."); return
        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.exists() or not checkpoint_path.is_dir():
            QMessageBox.warning(self, "Checkpoint Error", f"Checkpoint path does not exist or is not a directory:\n{checkpoint_path}"); return

        self.show_status("Loading models... This might take a while.", 0)
        self.progress_bar.setRange(0, 0); self.progress_bar.show(); QApplication.processEvents()
        try:
            (self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter) = load_models(checkpoint_path, status_callback=self.show_status)
            self.models_loaded = True
            QMessageBox.information(self, "Models Loaded", "Models have been loaded successfully.")
            self.show_status("Models loaded successfully. Ready to caption.", 5000)
        except Exception as e:
            self.models_loaded = False
            QMessageBox.critical(self, "Model Loading Error", f"An error occurred while loading models:\n{e}\n\nCheck console for details.")
            self.show_status(f"Model loading failed. Check console.", 0)
            print(f"--- Model Loading Error ---"); import traceback; traceback.print_exc(); print(f"--- End Error Traceback ---")
        finally:
            self.progress_bar.hide(); self.progress_bar.setRange(0, 100); self.update_button_states()

    def collect_parameters(self):
        return (self.caption_type_combo.currentText(), self.caption_length_combo.currentText(),
                [cb.text() for cb in self.extra_options_checkboxes if cb.isChecked()],
                self.name_input_line.text(), self.custom_prompt_text.toPlainText())

    def _confirm_overwrite(self, file_path: Path) -> bool:
        if file_path.exists():
            reply = QMessageBox.question(self, 'Confirm Overwrite', f"Caption file '{file_path.name}' already exists.\nOverwrite?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return reply == QMessageBox.Yes
        return True

    def _save_caption_to_file(self, image_path: Path, caption: str) -> bool:
        if not image_path: self.show_status("Error: No image path associated.", 5000); return False
        caption_file = image_path.with_suffix('.txt')
        mode = 'a' if self.append_checkbox.isChecked() else 'w'
        prefix = '\n' if mode == 'a' and caption_file.exists() and caption_file.stat().st_size > 0 else ''

        if mode == 'w' and caption_file.exists() and not self.overwrite_checkbox.isChecked():
            if not self._confirm_overwrite(caption_file):
                self.show_status(f"Skipped saving {image_path.name}.", 3000); return False
        try:
            with open(caption_file, mode, encoding='utf-8') as f: f.write(f"{prefix}{caption}")
            self.show_status(f"Caption {'appended to' if mode == 'a' else 'saved to'} {caption_file.name}", 4000); return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving caption for {image_path.name}:\n{e}")
            self.show_status(f"Error saving caption for {image_path.name}", 5000); print(f"Error saving caption to {caption_file}: {e}"); return False

    def _run_caption_generation(self, image_path: Path):
        if not self.models_loaded: QMessageBox.warning(self, "Models Not Loaded", "Please load models first."); return None
        if not image_path or not image_path.exists():
             QMessageBox.warning(self, "Image Not Found", f"Image file does not exist:\n{image_path}")
             self.show_status(f"Image not found: {image_path.name if image_path else 'None'}", 5000); return None

        self.show_status(f"Processing: {image_path.name}...", 0); QApplication.processEvents()
        params = self.collect_parameters()
        try: input_image = Image.open(image_path)
        except Exception as e:
            QMessageBox.critical(self, "Image Open Error", f"Failed to open {image_path.name}:\n{e}")
            self.show_status(f"Error opening {image_path.name}", 5000); print(f"Error opening image {image_path}: {e}"); return None
        try:
            prompt_str, caption = generate_caption(input_image, *params, self.clip_model, self.tokenizer, self.text_model, self.image_adapter)
            current_viewed_path = self.selected_image_path or self.single_image_path
            if image_path == current_viewed_path: self.generated_caption_text.setText(caption)
            if self._save_caption_to_file(image_path, caption): print(f"Caption generated and saved for {image_path.name}")
            else: print(f"Caption generated but NOT saved for {image_path.name}")
            return caption
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to process {image_path.name}:\n{e}\n\nCheck console.")
            self.show_status(f"Error processing {image_path.name}. Check console.", 0)
            print(f"--- Processing Error for {image_path.name} ---"); import traceback; traceback.print_exc(); print(f"--- End Error Traceback ---")
            current_viewed_path = self.selected_image_path or self.single_image_path
            if image_path == current_viewed_path: self.generated_caption_text.setText(f"Error generating caption. See console.")
            return None
        finally: QApplication.processEvents()

    def generate_captions_action(self):
        if not self.input_dir or not self.image_files: QMessageBox.warning(self, "No Images", "Select directory with images first."); return
        if not self.models_loaded: QMessageBox.warning(self, "Models Not Loaded", "Load models first."); return

        num_images = len(self.image_files)
        self.progress_bar.setRange(0, num_images); self.progress_bar.setValue(0); self.progress_bar.show()
        self.show_status(f"Starting batch captioning for {num_images} images...", 0)

        processed_count, error_count, skipped_explicitly = 0, 0, 0
        original_overwrite_state = self.overwrite_checkbox.isChecked() # Remember original state
        ask_all = False # Flag to check if user agreed to overwrite all

        # Pre-check for overwrites if needed
        files_to_confirm = []
        if not self.overwrite_checkbox.isChecked() and not self.append_checkbox.isChecked():
            files_to_confirm = [img.with_suffix('.txt').name for img in self.image_files if img.with_suffix('.txt').exists()]

        if files_to_confirm:
             reply = QMessageBox.question(self, 'Confirm Overwrite Multiple', f"{len(files_to_confirm)} existing caption file(s) found.\nOverwrite ALL existing files?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
             if reply == QMessageBox.Cancel: self.show_status("Batch cancelled.", 3000); self.progress_bar.hide(); return
             elif reply == QMessageBox.Yes: ask_all = True; self.overwrite_checkbox.setChecked(True) # Temporarily check it

        # Process images
        for i, image_path in enumerate(self.image_files):
            # Run generation. _save_caption_to_file handles individual confirmation if ask_all is False
            caption_result = self._run_caption_generation(image_path)

            # Track results (Approximate - relies on _save reporting skips)
            if caption_result is not None:
                processed_count += 1
            else:
                # If None, assume error unless status bar indicates skip (imperfect)
                if "Skipped saving" not in self.status_bar.currentMessage():
                     error_count += 1
                # No reliable way to count skips here without modifying _save return value

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        # Restore overwrite checkbox state if changed
        if ask_all: self.overwrite_checkbox.setChecked(original_overwrite_state)

        self.progress_bar.hide()
        final_message = f"Batch finished. {processed_count} captions generated/saved."
        if error_count > 0: final_message += f" {error_count} errors."
        # Cannot reliably report skips here
        QMessageBox.information(self, "Batch Complete", final_message)
        self.show_status(final_message, 10000)
        self.update_button_states()

    def caption_selected_image_action(self):
        if not self.selected_image_path: QMessageBox.warning(self, "No Image Selected", "Select image from list first."); return
        self._run_caption_generation(self.selected_image_path); self.update_button_states()

    def caption_single_image_action(self):
        if not self.single_image_path: QMessageBox.warning(self, "No Image Selected", "Select single image first."); return
        self._run_caption_generation(self.single_image_path); self.update_button_states()

    def save_edited_caption_action(self):
        edited_caption = self.generated_caption_text.toPlainText().strip()
        if not edited_caption: QMessageBox.warning(self, "Empty Caption", "Caption text is empty."); return
        current_image_path = self.selected_image_path or self.single_image_path
        if not current_image_path: QMessageBox.warning(self, "No Associated Image", "Select image first."); return
        self._save_caption_to_file(current_image_path, edited_caption)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current_path = None
        if self.selected_image_label.pixmap() and not self.selected_image_label.pixmap().isNull():
             current_path = self.selected_image_path or self.single_image_path
        if current_path and current_path.exists():
            try:
                pixmap = QPixmap(str(current_path))
                if not pixmap.isNull(): self.scale_and_set_pixmap(pixmap)
            except Exception as e: print(f"Error reloading pixmap on resize for {current_path}: {e}")
        elif not self.selected_image_label.text() or self.selected_image_label.text().startswith(("Cannot", "Error", "No image")):
             self.selected_image_label.clear(); self.selected_image_label.setText("No image selected")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) # Optional
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) # Optional
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Optional
    window = CaptionApp()
    window.show()
    sys.exit(app.exec_())