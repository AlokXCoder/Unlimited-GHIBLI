![Image](https://github.com/user-attachments/assets/f8d14c75-822b-4cb5-beb6-bac7718264df)

# 🥰 UNLIMITED FREE GHIBLI ART IMAGES METHOD

Turn any image into magical Ghibli-style art in just a few clicks — for FREE!

---

## 🚀 How to Generate Ghibli Images Using Your Python Code?

🧘‍♂️ Even you can make your own API!

### ⭐ Step-by-Step Guide

1. **Open this link in Chrome**  
   👉 [Google Colab](https://colab.research.google.com/) — A free platform to run Python code.

2. **Click on `+ New Notebook`**  
   Once clicked, a new window will open where you can code.

   🔧 **Tip for faster performance:**
   Before you run any code, go to the top menu and click on `Runtime` → `Change runtime type`, then set the hardware accelerator to `T4 GPU` and click `Save`.  
   🚀 This boosts the speed of image generation significantly.

4. **Copy & Paste the Required Code**  
   Paste the setup code in the `Start coding or generate with AI` section, followed by the main image generation code.

5. **Click the Run Icon**  
   ⏱ It only takes 2 seconds to start running.

6. **Upload Your Image**  
   - 🧿 Wait for the "Upload your image" box to appear.  
   - 🖥 Choose your file to upload.

7. **Select Stylization Strength**  
   💡 Recommended range: `0.5 – 0.6`  
   🔸 Best results at `0.55`.

8. **Wait 1–2 Minutes**  
   ✔️ Your Ghibli-style image will be automatically generated and downloaded!

---

## 🤔 Why Use This?

🥷 **Generate unlimited** Ghibli images for free — no subscription, no limits, just creativity.

---

Feel free to star ⭐ this repo if you found it helpful!


## 🌐 Source Code
```python
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverSinglestepScheduler
import torch
from PIL import Image, ImageOps
import io
from google.colab import files
import matplotlib.pyplot as plt
import time
import numpy as np

# Model Loader (Optimized)
def load_model():
    model_id = "nitrosocke/Ghibli-Diffusion"

    print("🔍 Loading Precision Ghibli Model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    # Precision scheduler
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Memory optimization
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    print("✅ Model Ready")
    return pipe

# Generation with Aspect Ratio Preservation
def generate_ghibli(image, pipe, strength=0.55):
    # Convert to RGB and pad to divisible by 8
    image = image.convert("RGB")
    w, h = image.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Dynamic prompt based on image content
    is_portrait = h > w
    prompt = (
        "Transform the given image into the distinctive hand-drawn animation style of Studio Ghibli. "
        "Preserve the original composition, characters, and scene elements while applying the following stylistic enhancements: "
        "Use soft, fluid lines and a dreamy, pastel color palette dominated by soft greens, blues, and warm earth tones. "
        "Enhance natural elements such as foliage, water, and light to create a whimsical, enchanted atmosphere. "
        "For any characters present, redesign them with expressive, large eyes, simplified yet charming features, "
        "and ensure their anatomy is correct with no extra limbs, distorted faces, or incorrect proportions. "
        "If the image contains landscapes or scenes without characters, subtly incorporate fantastical elements "
        "like glowing lights, magical creatures, or other Ghibli-esque details in the background. "
        "The final image should evoke a sense of nostalgia, peace, and wonder, characteristic of Ghibli's animation, "
        "and should appear as if it were a frame from one of their films, with a hand-painted, non-photorealistic look."
    )

    negative_prompt = (
        "blurry, deformed, bad anatomy, extra limbs, distorted faces, incorrect proportions, text, watermark, "
        "oversaturated, 3D render, cartoonish, low quality"
    )

    print("🖌️ Creating Ghibli magic...")
    start = time.time()

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=40,
        generator=torch.Generator(device="cuda").manual_seed(int(time.time()))
    ).images[0]

    # Restore original dimensions
    result = result.resize((w, h), Image.LANCZOS)
    print(f"⏱️ Completed in {(time.time()-start)/60:.1f} min")
    return result

# Main Execution with Display Fixes
try:
    # Check GPU
    assert torch.cuda.is_available(), "🚨 Enable GPU: Runtime → Change runtime type → T4 GPU"
    pipe = load_model()

    # Upload with immediate display
    print("⬆️ Upload image (any size/format):")
    uploaded = files.upload()
    if not uploaded:
        raise Exception("No image uploaded")

    file_name = list(uploaded.keys())[0]
    original = Image.open(io.BytesIO(uploaded[file_name]))

    # Display upload immediately
    print("\n🔍 Uploaded Image Preview:")
    plt.figure(figsize=(10, 6))
    plt.imshow(original)
    plt.axis('off')
    plt.title("Your Original Image", pad=20)
    plt.show()

    # Strength input
    while True:
        try:
            strength = float(input("💡 Stylization strength (0.5-0.6 recommended): "))
            if 0.4 <= strength <= 0.7:
                break
            print("Please enter 0.4-0.7")
        except:
            print("Numbers only")

    # Generate and display
    print("\n🎨 Generating Ghibli version...")
    ghibli_img = generate_ghibli(original, pipe, strength)

    # Side-by-side comparison
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original", pad=10)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ghibli_img)
    plt.title("Ghibli Version", pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save
    output_name = f"Ghibli_{file_name.split('.')[0]}.png"
    ghibli_img.save(output_name, quality=95)
    files.download(output_name)
    print(f"📥 Saved: {output_name}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("🔧 Try: 1) Restart runtime 2) Check GPU 3) Smaller image")
```

## LICENSE
[MIT LICENSE](LICENSE)