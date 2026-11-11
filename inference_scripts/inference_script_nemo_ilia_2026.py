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

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
# Force the math kernel to be the priority
torch.backends.cuda.enable_math_sdp(True)

os.environ["HIP_VISIBLE_DEVICES"] = "0"


# model_id = "mistralai/Mistral-Nemo-Instruct-2407"
model_id = "mistralai/Mistral-Nemo-Instruct-2407"

# --- 2. Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    # use_fast=True,
    padding_side="left",
    fix_mistral_regex=True
    )

tokenizer.pad_token = tokenizer.eos_token 

# max_memory = {0: "60GiB", "cpu": "128GiB"}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # device_map='auto',
    device_map={'': 0},
    # max_memory=max_memory,
    low_cpu_mem_usage=False,
    #offload_folder="offload" 
    attn_implementation="sdpa"
)

model.generation_config.pad_token_id = tokenizer.eos_token_id

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

# --- 3. Processing Loop ---
results = []
data_to_label = pd.read_csv("/home/roportot/gender_labeling_ilia/inv_consistente_no_xpac_fran_tt.csv")

output_path = "/home/roportot/gender_labeling_ilia/classification_nemo_ilia_2026_new_prompt.csv"

unique_data = data_to_label.drop_duplicates(["author_name", "country_code"]).copy() # Use .copy() to avoid SettingWithCopyWarning
# unique_data = unique_data[0:2000]

for i, row in tqdm(unique_data.iterrows(), total=unique_data.shape[0], desc="Nemo Labeling"):
    name = row["author_name"]
    country = str(row["country_code"])

    try:
        # Prepare Chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[NOMBRE] {name} [PAÍS] {country}"} # Cap text to avoid context blowup
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
                max_new_tokens=30, 
                do_sample=False, 
                max_length=512,
                use_cache=True, #tiny cache row by row
                # pad_token_id=tokenizer.eos_token_id, 
                #temperature=0.0
            )

        generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip().lower()
        
        # if "mujer" in response: 
        #     response = "mujer"
        # elif "hombre" in response: 
        #     response = "hombre"
        # else: 
        #     response = "desconocido"

        print(f"{name}: ", generated_text)

        results.append(generated_text)

        # 4. SAFE MEMORY RELEASE
        del inputs, outputs  # Remove references to heavy tensors
        if i % 30 == 0:
            torch.cuda.empty_cache()
            gc.collect()     # Clear Python garbage without touching GPU cache directly

    except Exception as e:
        print(f"Error on {name}: {e}")
        results.append("error")

# --- 5. Save ---
unique_data["gender_pred"] = results
unique_data.to_csv(output_path, index=False)
print(f"Classification done. Results saved.")
