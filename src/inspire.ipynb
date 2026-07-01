{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9609b1af",
   "metadata": {},
   "source": [
    "# INSPIRE DNN Mortality Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2dc11f5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check GPU\n",
    "import torch\n",
    "print('CUDA available:', torch.cuda.is_available())\n",
    "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\n",
    "# If False: Runtime -> Change runtime type -> T4 GPU -> Save"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb8c0538",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone repo and move into src/\n",
    "!git clone https://github.com/thrisharajkumar/inspire-analysis-thrisha.git\n",
    "import os\n",
    "os.chdir('/content/inspire-analysis-thrisha/src')\n",
    "print('Working directory:', os.getcwd())\n",
    "print('Files:', os.listdir('.'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d435931c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Upload your inspire_subjects_small.zip\n",
    "# Select inspire_subjects_small.zip when the file picker opens\n",
    "from google.colab import files\n",
    "uploaded = files.upload()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea8f04ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract the zip\n",
    "import zipfile, os\n",
    "\n",
    "zip_path = '/content/inspire-analysis-thrisha/src/inspire_subjects_small.zip'\n",
    "extract_dir = '/content/inspire_subjects_small'\n",
    "\n",
    "with zipfile.ZipFile(zip_path, 'r') as z:\n",
    "    z.extractall(extract_dir)\n",
    "\n",
    "# Handle nested folder (zip contains inspire_subjects_small/)\n",
    "contents = os.listdir(extract_dir)\n",
    "print('Top level contents:', contents)\n",
    "if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):\n",
    "    extract_dir = os.path.join(extract_dir, contents[0])\n",
    "    print('One level deeper:', os.listdir(extract_dir))\n",
    "\n",
    "# Confirm survived/died structure\n",
    "print('survived:', len(os.listdir(os.path.join(extract_dir, 'survived'))), 'files')\n",
    "print('died:    ', len(os.listdir(os.path.join(extract_dir, 'died'))), 'files')\n",
    "print('DATA PATH =', extract_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6f7688f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pull latest code from GitHub\n",
    "# (gets all the changes you pushed from VS Code)\n",
    "!git pull\n",
    "print('Files now:')\n",
    "print(os.listdir('.'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d9d2ede",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the pipeline\n",
    "# Expected output:\n",
    "#   loaded 30 subjects (10 died, 20 survived)\n",
    "#   Autoencoder Epoch 1/10 ... loss drops each epoch\n",
    "#   Classifier Epoch 1/20 ... loss drops each epoch  \n",
    "#   AUROC = 0.8+\n",
    "!python dnn_mortality_pipeline.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb0d57c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display output plots\n",
    "from IPython.display import Image, display\n",
    "import os\n",
    "\n",
    "for fname in ['embeddings.png', 'auroc.png', 'auprc.png']:\n",
    "    path = f'/content/inspire-analysis-thrisha/src/{fname}'\n",
    "    if os.path.exists(path):\n",
    "        print(f'--- {fname} ---')\n",
    "        display(Image(path))\n",
    "    else:\n",
    "        print(f'MISSING: {fname} — run Cell 7 first')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10ea7dc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save output plots to your Google Drive \n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "import shutil, os\n",
    "save_dir = '/content/drive/MyDrive/INSPIRE_results'\n",
    "os.makedirs(save_dir, exist_ok=True)\n",
    "\n",
    "for fname in ['embeddings.png', 'auroc.png', 'auprc.png']:\n",
    "    src = f'/content/inspire-analysis-thrisha/src/{fname}'\n",
    "    if os.path.exists(src):\n",
    "        shutil.copy(src, os.path.join(save_dir, fname))\n",
    "        print(f'Saved {fname} to Drive')\n",
    "    else:\n",
    "        print(f'MISSING: {fname}')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
