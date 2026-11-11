
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
from tqdm import tqdm
import gc

# --- 1. Setup ---
os.environ["HF_HOME"] = "/home/roportot/archive/huggingface_cache"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.10"
os.environ["HF_TOKEN"] = "your_token" 

# Disable the specific backend that often causes 'busy waiting' on MI210
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
# Force the math kernel to be the priority
torch.backends.cuda.enable_math_sdp(True)

os.environ["HIP_VISIBLE_DEVICES"] = "0"

model_id = "google/gemma-3-12b-it"

# --- 2. Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    padding_side="left"
)

max_memory = {0: "56GiB", "cpu": "128GiB"}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    device_map={'': 0},
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

# system_prompt = (
#     """
#     Eres un experto en onomástica. \n  
#     Tu tarea consiste en clasificar nombres por el género al que tienen la mayor probabilidad de estar asociados.
#     Recibirás también un país al que está asociado el nombre de esta persona, puesto que ha trabajado en una institución
#     perteneciente a dicho país.

#     Reglas:
#     1- El formato en que recibirás la información tiene la siguiente estructura: [NOMBRE] nombre [PAÍS] país \n
#     2- Etiquetas: "mujer", "hombre" o "desconocido". Responde ÚNICAMENTE con la etiqueta en minúscula. \n
#     3- Si encuentras un nombre ambiguo que se utilice de forma neutra sin estar asociado predominantemente 
#     a un género específico, o sus datos son insuficientes, como nombres abreviados, etiquétalo como "desconocido"
# """
# )

system_prompt = (
    """
    Eres un experto en onomástica (estudio de los nombres propios).
    Tu tarea es inferir el género más probable asociado a un nombre de persona, considerando también el país proporcionado como contexto cultural.\n\n 
    Entrada/INPUT:\n
    Recibirás un texto con el siguiente formato:\n
    ---- [NOMBRE] nombre completo [PAÍS] país \n\n 
    Instrucciones:\n
    Analiza el nombre y determina el género al que está más comúnmente asociado en el contexto del país indicado. No se puede definir el género a partir del apellido, es solo a partir de su primer nombre y/o segundo nombre.
    Debes responder únicamente con una de las siguientes etiquetas (en minúsculas y sin explicaciones adicionales):\n
    mujer \n
    hombre \n
    desconocido_sigla: cuando el nombre sea una inicial, abreviatura o no contenga información suficiente (por ejemplo: "H. Pérez" o solo letras sueltas). \n
    desconocido_ambiguedad: cuando el nombre sea genuinamente ambiguo o de uso común en más de un género sin predominancia clara. \n\n
    Reglas adicionales:\n
    No incluyas explicaciones ni texto adicional, solo la etiqueta.\n
    Prioriza el uso cultural del país proporcionado cuando sea relevante.\n
    Ejemplos:\n 
    hombre : Marc, Joan, Pau, Alejandro, José, Thiago, Santiago, Benjamín, Matías, Simón, Juan.\n 
    mujer :  Paula, Lucía, María, Marta, Carla, María José, Yusimí, Yoanis, Yanet.
    """
)

# 4- Entrega, después de la etiqueta de género, un número entre 1, 2 y 3, donde 1 es poca certeza, 2 es media certeza y
# 3 es mucha certeza, basándote en la confianza que tengas de que dicho nombre corresponda con el género que inferiste.

# --- 3. Processing Loop ---
input_path = "/home/roportot/gender_labeling_ilia/inv_consistente_no_xpac_fran_tt.csv"
output_path = "/home/roportot/gender_labeling_ilia/classification_gemma_ilia_2026_new_prompt.csv"

data_to_label = pd.read_csv(input_path)
unique_data = data_to_label.drop_duplicates(["author_name", "country_code"])
# unique_data = unique_data[0:1000]
results = []

for i, row in tqdm(unique_data.iterrows(), total=len(unique_data), desc="Gemma Labeling"):
    name = str(row["author_name"])
    country = str(row["country_code"])

    try:
        # Gemma 3 uses a specific chat template structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[NOMBRE] {name} [PAÍS] {country}"}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=30, # Gemma is precise, but keep it short
                do_sample=False, 
                # Gemma 3 doesn't strictly need temperature=0 if do_sample is False
            )

        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip().lower()
        
        # Basic validation to keep data clean
        # if "mujer" in generated_text:
        #     clean_res = "mujer"
        # elif "hombre" in generated_text:
        #     clean_res = "hombre"
        # else:
        #     clean_res = "desconocido"

        results.append(generated_text)

        # 4. Memory Management
        del inputs, outputs
        if i % 30 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"Error on {name}: {e}")
        results.append("error")

# --- 5. Save ---
unique_data["gender_pred"] = results
unique_data.to_csv(output_path, index=False)
print(f"Classification done. Results saved.")
