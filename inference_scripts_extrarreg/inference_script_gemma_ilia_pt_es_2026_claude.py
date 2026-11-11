import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import csv
import pandas as pd
from tqdm import tqdm
import gc

# --- 1. Setup ---
os.environ["HF_HOME"] = "/home/roportot/archive/huggingface_cache"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.10"
os.environ["HF_TOKEN"] = "your_token"

# Disable the specific backend that often causes 'busy waiting' on MI210
#torch.backends.cuda.enable_flash_sdp(False)
#torch.backends.cuda.enable_mem_efficient_sdp(False)
# Force the math kernel to be the priority
#torch.backends.cuda.enable_math_sdp(True)

os.environ["HIP_VISIBLE_DEVICES"] = "0"

model_id = "google/gemma-3-12b-it"
#model_id = "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"

# --- Batch size: start conservative, raise if VRAM allows ---
# 12B in bf16 is ~24GB of weights; MI210 has 64GB, so there's plenty of
# room for activations at a decent batch size. Start here and watch
# `rocm-smi` / `torch.cuda.max_memory_allocated()` to tune it up.
BATCH_SIZE = 96
SAVE_EVERY_N_BATCHES = 1  # flush to disk after every batch by default

# --- 2. Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

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
    hombre : Marc, Joan, Pau, Aleksandr, Sergei, Alejandro, José, Thiago, Santiago, Benjamín, Matías, Simón, Juan, Karl, Matteo, Wilhelm, Otto, Peter, Peeter, Kadri, Heinrich .\n 
    mujer :  Paula, Lucía, María, Marta, Sofia, Mia, Aron, Olga, Irina, Svetlana, Carla, María José, Yusimí, Yoanis, Yanet, Emma, Emilia, Anna, Greta, Wilma.
    """
)

# --- 3. Load input data ---
input_path = "/home/roportot/gender_labeling_reference_countries/inv_consistente_data.csv"
output_path = "/home/roportot/gender_labeling_reference_countries/classification_sg_gemma_reference_2026_new_prompt.csv"

data_to_label = pd.read_csv(input_path)
unique_data = data_to_label.drop_duplicates(["author_name", "country_code"])
unique_data = data_to_label[data_to_label["country_code"].isin(["SG"])]
unique_data = unique_data.reset_index(drop=True)

fieldnames = list(unique_data.columns) + ["gender_pred"]

# --- 4. Resume support: figure out what's already been labeled ---
already_done = set()
file_exists = os.path.isfile(output_path)

if file_exists:
    try:
        existing = pd.read_csv(output_path)
        already_done = set(zip(existing["author_name"].astype(str), existing["country_code"].astype(str)))
        print(f"Resuming: {len(already_done)} rows already labeled, skipping those.")
    except Exception as e:
        print(f"Could not read existing output file ({e}); starting fresh.")
        file_exists = False

out_f = open(output_path, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(out_f, fieldnames=fieldnames)
if not file_exists or os.path.getsize(output_path) == 0:
    writer.writeheader()
    out_f.flush()

# --- 5. Build the list of rows still needing labels ---
pending_rows = [
    row for _, row in unique_data.iterrows()
    if (str(row["author_name"]), str(row["country_code"])) not in already_done
]
print(f"{len(pending_rows)} rows to label (batch size {BATCH_SIZE}).")

# --- 6. Batched processing loop ---
processed_count = 0
batch_count = 0

for batch_start in tqdm(range(0, len(pending_rows), BATCH_SIZE), desc="Gemma Labeling (batched)"):
    batch_rows = pending_rows[batch_start: batch_start + BATCH_SIZE]

    prompts = []
    for row in batch_rows:
        name = str(row["author_name"])
        country = str(row["country_code"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[NOMBRE] {name} [PAÍS] {country}"}
        ]
        # tokenize=False here -> we batch-tokenize all prompts together below
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # chat template already added special tokens
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,   # labels are short; trimmed from 30
                do_sample=False,
            )

        # Because of left-padding, every sequence's real input ends at
        # the same position (input_len), so the generated continuation
        # for every row starts at the same column.
        generated_only = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

        for row, text in zip(batch_rows, decoded):
            row_dict = row.to_dict()
            row_dict["gender_pred"] = text.strip().lower()
            writer.writerow(row_dict)
            processed_count += 1

        del inputs, outputs, generated_only

    except Exception as e:
        name_list = [str(r["author_name"]) for r in batch_rows]
        print(f"Error on batch starting at {batch_start} ({name_list}): {e}")
        for row in batch_rows:
            row_dict = row.to_dict()
            row_dict["gender_pred"] = "error"
            writer.writerow(row_dict)
            processed_count += 1

    batch_count += 1

    if batch_count % SAVE_EVERY_N_BATCHES == 0:
        out_f.flush()
        os.fsync(out_f.fileno())

    if batch_count % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()

out_f.flush()
os.fsync(out_f.fileno())
out_f.close()
print(f"Classification done. {processed_count} new rows labeled and saved to {output_path}")
