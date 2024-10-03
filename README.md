# joy-caption-alpha-two  gui-mod
joy-caption-alpha-two  gui mod

Installation-

Use one click installer-

Just download and click on the one click installer.
Download the adapter_config.json and copy abd replace it in  joy-caption-alpha-twoc\cgrkzexw-599808\text_model







Manual install:-


git clone https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two

cd joy-caption-alpha-two

python -m venv venv

venv\Scripts\activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pip install protobuf

pip install --upgrade PyQt5

Running:- 

activate venv

python caption_gui.py
