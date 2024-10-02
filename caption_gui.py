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
)
from PIL import Image
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List
from pathlib import Path

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
    QSplitter,
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt

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

class CaptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Captioning Application")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()

        # Initialize model variables
        self.clip_processor = None
        self.clip_model = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left panel for parameters
        left_panel = QVBoxLayout()

        # Input image selection
        self.input_image_button = QPushButton("Select Input Image")
        self.input_image_button.clicked.connect(self.select_input_image)
        self.input_image_label = QLabel("No image selected")
        left_panel.addWidget(self.input_image_button)
        left_panel.addWidget(self.input_image_label)

        # Input directory selection
        self.input_dir_button = QPushButton("Select Input Directory")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        self.input_dir_label = QLabel("No directory selected")
        left_panel.addWidget(self.input_dir_button)
        left_panel.addWidget(self.input_dir_label)

        # Caption Type
        self.caption_type_combo = QComboBox()
        self.caption_type_combo.addItems(CAPTION_TYPE_MAP.keys())
        self.caption_type_combo.setCurrentText("Descriptive")
        left_panel.addWidget(QLabel("Caption Type:"))
        left_panel.addWidget(self.caption_type_combo)

        # Caption Length
        self.caption_length_combo = QComboBox()
        self.caption_length_combo.addItems(CAPTION_LENGTH_CHOICES)
        self.caption_length_combo.setCurrentText("long")
        left_panel.addWidget(QLabel("Caption Length:"))
        left_panel.addWidget(self.caption_length_combo)

        # Extra Options
        left_panel.addWidget(QLabel("Extra Options:"))
        self.extra_options_checkboxes = []
        for option in EXTRA_OPTIONS_LIST:
            checkbox = QCheckBox(option)
            self.extra_options_checkboxes.append(checkbox)
            left_panel.addWidget(checkbox)

        # Name Input
        self.name_input_line = QLineEdit()
        left_panel.addWidget(QLabel("Person/Character Name (if applicable):"))
        left_panel.addWidget(self.name_input_line)

        # Custom Prompt
        self.custom_prompt_text = QTextEdit()
        left_panel.addWidget(QLabel("Custom Prompt (optional):"))
        left_panel.addWidget(self.custom_prompt_text)

        # Checkpoint Path
        self.checkpoint_path_line = QLineEdit()
        self.checkpoint_path_line.setText("cgrkzexw-599808")
        left_panel.addWidget(QLabel("Checkpoint Path:"))
        left_panel.addWidget(self.checkpoint_path_line)

        # Load Models Button
        self.load_models_button = QPushButton("Load Models")
        self.load_models_button.clicked.connect(self.load_models)
        left_panel.addWidget(self.load_models_button)

        # Run Button
        self.run_button = QPushButton("Generate Captions")
        self.run_button.clicked.connect(self.generate_captions)
        left_panel.addWidget(self.run_button)

        # Right panel for image display and captions
        right_panel = QVBoxLayout()

        # List widget for images
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        right_panel.addWidget(QLabel("Images:"))
        right_panel.addWidget(self.image_list_widget)

        # Display area for selected image
        self.selected_image_label = QLabel()
        self.selected_image_label.setFixedSize(400, 400)  # Adjust size as needed
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(QLabel("Selected Image:"))
        right_panel.addWidget(self.selected_image_label)

        # Button to generate caption for selected image
        self.caption_single_button = QPushButton("Generate Caption for Selected Image")
        self.caption_single_button.clicked.connect(self.generate_caption_for_selected_image)
        right_panel.addWidget(self.caption_single_button)

        # Adjusting the layout proportions
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        self.setLayout(main_layout)

    def select_input_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)", options=options)
        if file_name:
            self.input_image_path = Path(file_name)
            self.input_image_label.setText(str(self.input_image_path))
            # Display the selected image
            pixmap = QPixmap(str(self.input_image_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.selected_image_label.width(), self.selected_image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.selected_image_label.setPixmap(scaled_pixmap)
            # Clear any previously selected directory and image list
            self.input_dir = None
            self.input_dir_label.setText("No directory selected")
            self.image_list_widget.clear()
        else:
            self.input_image_label.setText("No image selected")
            self.input_image_path = None
            self.selected_image_label.clear()

    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = Path(directory)
            self.input_dir_label.setText(str(self.input_dir))
            self.load_images()
            # Clear any previously selected image
            self.input_image_path = None
            self.input_image_label.setText("No image selected")
            self.selected_image_label.clear()
        else:
            self.input_dir_label.setText("No directory selected")
            self.input_dir = None

    def load_images(self):
        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        # Collect all image files in the directory
        self.image_files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No image files found in the selected directory.")
            return

        self.image_list_widget.clear()
        for image_path in self.image_files:
            item = QListWidgetItem(str(image_path.name))
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                # Increase the thumbnail size
                scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon = QIcon(scaled_pixmap)
                item.setIcon(icon)
            self.image_list_widget.addItem(item)

    def display_selected_image(self, item):
        # Get the selected image path
        image_name = item.text()
        image_path = self.input_dir / image_name
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.selected_image_label.width(), self.selected_image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.selected_image_label.setPixmap(scaled_pixmap)

    def load_models(self):
        checkpoint_path = Path(self.checkpoint_path_line.text())
        if not checkpoint_path.exists():
            QMessageBox.warning(self, "Checkpoint Error", f"Checkpoint path does not exist: {checkpoint_path}")
            return

        try:
            (
                self.clip_processor,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            ) = load_models(checkpoint_path)
            QMessageBox.information(self, "Models Loaded", "Models have been loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"An error occurred while loading models: {e}")

    def generate_captions(self):
        # Check if input image is selected
        if hasattr(self, 'input_image_path') and self.input_image_path is not None:
            image_paths = [self.input_image_path]
        elif hasattr(self, 'image_files') and self.image_files:
            image_paths = self.image_files
        else:
            QMessageBox.warning(self, "No Images", "Please select an image or directory containing images.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        # Collect parameters
        caption_type = self.caption_type_combo.currentText()
        caption_length = self.caption_length_combo.currentText()
        extra_options = [checkbox.text() for checkbox in self.extra_options_checkboxes if checkbox.isChecked()]
        name_input = self.name_input_line.text()
        custom_prompt = self.custom_prompt_text.toPlainText()

        # Process each image
        for image_path in image_paths:
            print(f"\nProcessing image: {image_path}")
            input_image = Image.open(image_path)

            try:
                prompt_str, caption = generate_caption(
                    input_image,
                    caption_type,
                    caption_length,
                    extra_options,
                    name_input,
                    custom_prompt,
                    self.clip_model,
                    self.tokenizer,
                    self.text_model,
                    self.image_adapter,
                )

                # Save the caption in a text file with the same name as the image
                caption_file = image_path.with_suffix('.txt')
                with open(caption_file, 'w', encoding='utf-8') as f:
                    # Just write the caption
                    f.write(f"{caption}\n")

                print(f"Caption saved to {caption_file}")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        QMessageBox.information(self, "Captions Generated", "Captions have been generated and saved.")

    def generate_caption_for_selected_image(self):
        # Get the selected item
        selected_items = self.image_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list.")
            return
        item = selected_items[0]
        image_name = item.text()
        image_path = self.input_dir / image_name

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        # Collect parameters
        caption_type = self.caption_type_combo.currentText()
        caption_length = self.caption_length_combo.currentText()
        extra_options = [checkbox.text() for checkbox in self.extra_options_checkboxes if checkbox.isChecked()]
        name_input = self.name_input_line.text()
        custom_prompt = self.custom_prompt_text.toPlainText()

        try:
            input_image = Image.open(image_path)

            prompt_str, caption = generate_caption(
                input_image,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            )

            # Save the caption in a text file with the same name as the image
            caption_file = image_path.with_suffix('.txt')
            with open(caption_file, 'w', encoding='utf-8') as f:
                # Just write the caption
                f.write(f"{caption}\n")

            QMessageBox.information(self, "Caption Generated", f"Caption has been generated and saved for {image_name}.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while generating the caption: {e}")
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptionApp()
    window.show()
    sys.exit(app.exec_())
