---
license: mit
base_model: stabilityai/stable-diffusion-2
---
### horror-spooky on Stable Diffusion
This is the `<horror-spooky>` concept taught to Stable Diffusion via Textual Inversion. You can load this concept into the [Stable Conceptualizer](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb) notebook. You can also train your own concepts and load them into the concept libraries using [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb).

Here is the new concept you will be able to use as a `style`:
    ![<horror-spooky> 0](https://huggingface.co/sd-concepts-library/horror-spooky/resolve/main/concept_images/0.jpeg)
    ![<horror-spooky> 1](https://huggingface.co/sd-concepts-library/horror-spooky/resolve/main/concept_images/1.jpeg)
    ![<horror-spooky> 2](https://huggingface.co/sd-concepts-library/horror-spooky/resolve/main/concept_images/2.jpeg)
    ![<horror-spooky> 3](https://huggingface.co/sd-concepts-library/horror-spooky/resolve/main/concept_images/3.jpeg)
    ![<horror-spooky> 4](https://huggingface.co/sd-concepts-library/horror-spooky/resolve/main/concept_images/4.jpeg)
    
    "# sd-concept-horror" 

**Instructions to run the code**

1. Install Python 3.11 on your computer.

2. Use the -- python3.11 -m venv venv -- command to create a virtual environment

3. Use Git Bash to activate it: -- source/venv/Scripts/activate

4. Install the following tools/libraries from command prompt:
   Pip
   PyTorch
   Slugify
   Use command -- pip install -q --upgrade transformers==4.25.1 diffusers ftfy accelerate --
   Use command -- pip install accelerate --
   -- pip install numpy --
   -- pip install scipy --
   -- pip install requests --

5. Make sure to install CUDA and have it set up on your computer.

6. To run to training of the sd-concept stylistic model run the TxtInv.py file from the command line. This took my NVIDIA RTX 4070 about 8-9 hours to run locally, so be prepared to leave your code to run.

8. To create the learned embedding for the new trained concept, run the saveToHuggingFace.py file. (Note: this does not actually save the concept to hugging face, it simply generates the sd-concept-output folder to be later uploaded.

9. Finally, to run the file that generates most of the testing and output, run the SimpleDiffusion.py file, which culminates in the final spooky swamp forest monster output.
   
   

