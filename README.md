# joy-caption-alpha-two--cli-mod-and-gui-mod
joy-caption-alpha-two -cli mod and gui mod

Installation-
git clone https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two
cd joy-caption-alpha-two
python -m venv venv
venv\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install protobuf


Use app.py for cli


Parameters:

input_image: Path to the input image (for single image processing).
--input_dir: Path to the directory containing images (for batch processing).
--output_file: Path to the output file where captions will be saved (default is captions.txt).
--caption_type: Type of caption to generate.
--caption_length: Length of the caption.
--extra_options: Indices of extra options to customize the caption (use --list_extra_options to see options).
--name_input: Name of the person/character (used if an extra option requires it).
--custom_prompt: Custom prompt to override all other settings.
--checkpoint_path: Path to the model checkpoint directory (default is cgrkzexw-599808).
--list_extra_options: If specified, the script will list available extra options and exit.
To list available extra options:
python app.py --list_extra_options
