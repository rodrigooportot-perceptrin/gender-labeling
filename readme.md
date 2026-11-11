El subindicador de proporción de autoras en IA busca medir la representación de mujeres en  investigación académica relacionada a inteligencia artificial. Este subindicador representa el porcentaje promedio de mujeres investigadoras en IA respecto al total de investigadores de la misma disciplina dentro de los últimos 5 años completos (2021 a 2025). La fuente de datos corresponde a OpenAlex.

La formulación del subindicador, para cada país, es la siguiente:

```math
\frac{1}{|\text{años}|} \sum_{i \in \text{años}} \frac{m_i}{t_i} \times 100 \;,\qquad \text{años} = \{2021, 2022, 2023, 2024, 2025\}
```

donde m y t corresponden al número de mujeres y total de investigadores respectivamente, para el año i. La división por la magnitud del conjunto de años equivale a 5, entregando el promedio de estas fracciones. Luego se amplifica este resultado por 100 para su representación porcentual.

Cabe señalar que el género asociado a una persona en un trabajo académico es el percibido exclusivamente a partir del nombre textual de esta y la región donde se sitúa la investigación, que puede diferir con la identidad declarada por la misma. Esta brecha se sitúa fuera del alcance del proyecto.

La principal diferencia en el etiquetado respecto a las versiones anteriores del índice es el reemplazo del modelo computacional que ejecuta la clasificación de los nombres. Para esta versión, se optó por utilizar modelos grandes de lenguaje (LLMs) por sobre los servicios de etiquetado disponibles en internet empleados anteriormente (Namsor y Genderize.io).

Esta evolución se sostiene sobre el estudio de literatura del estado del arte en bibliometría de clasificación de género con modelos computacionales, así como un mayor control y rapidez sobre el proceso de etiquetado al disponer CENIA de GPUs donde correr tareas de inferencia con LLMs, ejecutando múltiples iteraciones de este experimento eficientemente. Estos son los puntos de mayor importancia: una mayor reproducibilidad y una menor opacidad en la configuración del proceso al comparar con los servicios comerciales en internet y sus limitaciones.

Para esta versión del índice, la formulación del subindicador de proporción de autoras en IA evolucionó con modificaciones en la metodología del etiquetado de género. Esta nueva clasificación contiene los siguientes elementos:

- mujer
- hombre
- desconocido_ambiguo
- desconocido_sigla

Si bien las dos primeras etiquetas son autodescriptivas, la tercera y cuarta son incluidas para casos donde no es posible determinar con certeza suficiente el presunto género de una persona: desconocido por ambigüedad existe para nombres unisex o de género neutro, mientras que desconocido por sigla se asigna a nombres abreviados en iniciales que no pueden ser extendidos para discernir un posible género. Estas categorías de abstención entregan flexibilidad en casos de incertidumbre.

[You, Z. et al (2024)](https://aclanthology.org/2024.gebnlp-1.16.pdf) utilizan una categoría neutra en identificación de género mediante nombre, puesto que los LLMs sujetos a prueba en su estudio tienen dificultades para clasificar nombres unisex, a la vez que distintos modelos difieren en facilidad para elegir la etiqueta neutra. También muestran que los intentos con five-shot prompting entregaron mejores resultados que zero-shot.

A su vez, advierten tomar precauciones al utilizar este tipo de clasificaciones con LLMs en tareas posteriores o downstream tasks. Aparte de indicar menor precisión para nombres neutros, existe un riesgo de sesgo en los modelos para clasificar nombres que no provienen del inglés. Para efectos del subindicador, esto se traduce en analizar manualmente los resultados de una muestra etiquetada por distintos modelos para evaluar sus desempeños.

Estas cuatro categorías están sujetas al contexto de cada país o región. Un nombre en un país específico puede estar más cercano a un género que a otro al compararlo con un lugar distinto. La región comparte principalmente el idioma español, mientras que en Brasil está presente el portugués, y en proporción menor también están los países de habla inglesa como Jamaica y Trinidad y Tobago. Para abordar esto, los LLMs reciben información adicional de entrada junto al nombre.

Dado que OpenAlex dispone, dentro de los metadatos de un trabajo, tanto el nombre como el apellido de una persona junto al código de país de la institución donde se produjo la investigación, esto se utiliza en conjunto para clasificar el género con la mayor certeza posible, considerando las restricciones ya mencionadas en las propiedades de estos datos y de los modelos, así como el gran volumen que abarcan.

Los LLMs pueden recibir información diversa dentro del prompt encargado de explicar la tarea de inferencia, siendo esto una ventaja relativa respecto a los campos limitados y respuestas rígidas que los servicios por APIs ofrecen. [Herrero-Solana et al. (2026)](https://zenodo.org/records/18610104) experimentaron con diversos modelos de lenguaje de pesos abiertos del tipo instruct para varias familias (Llama, Gemma, Phi, Mistral, Qwen, Deepseek, Yi) utilizando nombre y país de una persona como datos de entrada en el prompt, junto a tres posibles categorías de género (hombre, mujer y desconocido).

Los modelos elegidos variaron en tamaño entre los 270 millones y 70 mil millones de parámetros, donde muestran que el desempeño es aceptable alrededor de los 7 mil millones, con un rendimiento decreciente al superar el umbral de los 12-14 mil millones. La precisión utilizada para cargar el modelo  fue entre 8 y 16 bits, ejecutando inferencia local con zero-shot prompting.

En su diseño redactan un prompt asignando un rol de experto para ejecutar la tarea, añadiendo las posibles opciones a etiquetar, junto a un par de instrucciones que entregan contexto a los datos de entrada.

A partir de esto y lo señalado en el trabajo de You, Z. et al, para la construcción de este subindicador se prepara un prompt de características similares con múltiples ejemplos de nombres latinoamericanos y sus posibles géneros (few-shot prompting). Estos ejemplos suman información adicional a su conocimiento de nombres en inglés.

Este camino se basa en la técnica de in-context learning (ICL) donde un LLM recibe ejemplos aumentando el contexto, adaptando su inferencia en pos de un mejor desempeño para resolver la tarea. [Dong et al. (2024)](https://arxiv.org/pdf/2301.00234) explican que ICL es sensible a la selección de ejemplos y al orden de estos en el prompt. Tomando esto en cuenta, la estructura del prompt utilizado fue constante entre los modelos para los países de la región latinoamericana. Para el etiquetado extrarregional, véase su sección particular.

El prompt y los LLMs seleccionados fueron los siguientes:

[CUADRO PROMPT]

```python
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
    mujer :  Paula, Lucía, María, Marta, Carla, María José, Yusimí, Yoanis, Yanet.
    """
)
```


- [Gemma 3 12b Instruct](https://huggingface.co/google/gemma-3-12b-it)
- [Llama 3.1 8b Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Mistral Small 22b Instruct](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)
- [Mistral Nemo 12 b Instruct](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)

Este grupo de LLMs poseen características similares en tamaño, dentro del rango de los 8 y 22 mil millones de parámetros, todos del tipo instruct. Su selección se basa en la literatura descrita anteriormente según sus características técnicas. Estos modelos se cargan localmente con la librería de Python transformers de Hugging Face, plataforma que aloja y disponibiliza modelos de aprendizaje de máquinas, conjuntos de datos y aplicaciones en la red.

Los pesos de los modelos se cargaron en GPUs de alto rendimiento AMD Instinct MI210 con una memoria de 64 GB, utilizando librerías de Pytorch adaptadas al software ROCm compatible con el uso de esta marca de tarjetas. Una sola tarjeta tiene capacidad suficiente para almacenar los modelos elegidos sin cuantizar y todo lo requerido para su uso, como la ventana de contexto y el caché, incluso con el modelo de más parámetros de la lista (Mistral Small).

(Luego sesgos en ICL y modelos con dong et al y otro paper de clasificación de género)

Una dificultad importante presente en esta tarea es la ausencia de Ground Truth o verdad de campo al carecer de un conjunto de nombres etiquetados con género verificado. Esta es la principal razón por la que el género etiquetado es el percibido a partir de los datos (nombre y país de investigación). Esta limitación convierte la verificación manual de la clasificación hecha por los modelos en un ejercicio de concordancia entre estos y la percepción de la persona encargada de analizar los resultados.

Con el conjunto de nombres ya clasificados por los cuatro modelos, se toma una muestra de 1.000 elementos aleatorios (constante entre los cuatro conjuntos producidos) para revisar manualmente estas etiquetas y luego estimar el desempeño de los modelos siguiendo diversas métricas como exactitud, precisión, recall o F1.

Las definiciones de estás métricas se explican brevemente a continuación, considerando la clase c como una de las clases a medir, puesto que este es un problema de etiquetado con cuatro clases o categorías diferentes.
| Métrica | Definición | Fórmula |
|---|---|---|
| Exactitud o accuracy | Proporción de todas las filas correctamente clasificadas | $`\frac{\sum_c TP_c}{N}`$<br>\*TPc es True Positives de la clase c<br>\*N es tamaño del conjunto, en este caso de la muestra |
| Precisión | Proporción de filas etiquetadas con clase c que realmente pertenecen a dicha clase. | $`P_c = \frac{TP_c}{TP_c + FP_c}`$<br>\*TPc es True Positives de la clase c<br>\*FPc es False Positives de la clase c |
| Sensibilidad o recall | Proporción de filas etiquetadas con éxito por el modelo con clase c que realmente pertenecen a dicha clase. | $`R_c = \frac{TP_c}{TP_c + FN_c}`$<br>\*TPc es True Positives de la clase c<br>\*FNc es False Negatives de la clase c |
| F1 | Media armónica entre precisión y recall. | $`F1_c = \frac{2 P_c R_c}{P_c + R_c}`$ |
| Precisión Macro | Precisión promedio, cada clase pesa lo mismo. | $`\frac{1}{\lvert \text{Clases} \rvert} \sum_c P_c`$<br>\*Clases es el conjunto con las categorías de las clases, en este caso su magnitud equivale a 4 |
| Recall Macro | Recall promedio, cada clase pesa lo mismo. | $`\frac{1}{\lvert \text{Clases} \rvert} \sum_c R_c`$<br>\*Clases es el conjunto con las categorías de las clases, en este caso su magnitud equivale a 4 |
| F1 Macro | F1 promedio, cada clase pesa lo mismo. | $`\frac{1}{\lvert \text{Clases} \rvert} \sum_c F1_c`$<br>\*Clases es el conjunto con las categorías de las clases, en este caso su magnitud equivale a 4 |
| Soporte | Número de filas que pertenecen verdaderamente a la clase c. | $`S_c = TP_c + FN_c`$<br>\*TPc es True Positives de la clase c<br>\*FNc es False Negatives de la clase c |
| F1 Ponderado | F1 promedio ponderado en función del tamaño de cada clase en el conjunto. | $`\sum_c \frac{S_c}{N} F1_c`$<br>\*N es el tamaño del conjunto, en este caso la muestra |


El conjunto de datos disminuyó levemente su tamaño por una limpieza previa, quitando observaciones con datos erróneos o mal catalogados como nombres (ejemplos: “Cadernos de Pós-Graudação em Letras”, “Una Clase De Equivalencia”). Este ruido es complejo de abordar, puesto que proviene de la fuente original. Se propone una limpieza más profunda a partir de la siguiente edición del índice. A su vez, el análisis fue realizado por dos integrantes del equipo, con leves discrepancias en la clasificación de algunos nombres, disminuyendo también el total de filas a analizar por consenso. Todo esto redujo la dimensión de las muestras de datos a un total de 996 filas.

| Modelo / Métrica | Accuracy | Precisión Macro | Recall Macro | F1 Macro | F1 Ponderado |
|---|---|---|---|---|---|
| Gemma | 0,93 | 0,94 | 0,74 | 0,73 | 0,91 |
| Llama | 0,68 | 0,65 | 0,68 | 0,63 | 0,68 |
| Nemo | 0,89 | 0,76 | 0,90 | 0,80 | 0,90 |
| Small | 0,76 | 0,81 | 0,65 | 0,66 | 0,76 |


| Modelo / Métrica | Clase | Precisión | Recall | F1 | Soporte | Presencia porcentual de la categoría en la muestra |
|---|---|---|---|---|---|---|
| Gemma | hombre | 0.93 | 0.98 | 0.96 | 610 | 61,2% |
| | mujer | 0.94 | 0.97 | 0.96 | 268 | 26,9% |
| | desconocido_sigla | 0.89 | 0.94 | 0.91 | 66 | 6,6% |
| | desconocido_ambiguo | 1.00 | 0.06 | 0.11 | 52 | 5,2% |
| Llama | hombre | 1.00 | 0.55 | 0.71 | 610 | 61,2% |
| | mujer | 0.50 | 1.00 | 0.66 | 268 | 26,9% |
| | desconocido_sigla | 0.81 | 0.89 | 0.81 | 66 | 6,6% |
| | desconocido_ambiguo | 0.32 | 0.37 | 0.34 | 52 | 5,2% |
| Nemo | hombre | 1.00 | 0.89 | 0.94 | 610 | 61,2% |
| | mujer | 0.99 | 0.88 | 0.93 | 268 | 26,9% |
| | desconocido_sigla | 0.67 | 0.94 | 0.79 | 66 | 6,6% |
| | desconocido_ambiguo | 0.39 | 0.90 | 0.54 | 52 | 5,2% |
| Small | hombre | 1.00 | 0.78 | 0.83 | 610 | 61,2% |
| | mujer | 0.54 | 1.00 | 0.70 | 268 | 26,9% |
| | desconocido_sigla | 0.94 | 0.68 | 0.79 | 66 | 6,6% |
| | desconocido_ambiguo | 0.79 | 0.21 | 0.33 | 52 | 5,2% |

Gemma lidera en 3 de 5 métricas Macro, mientras que Nemo alcanza mejores puntajes en Recall Macro y F1 Macro. Esto sugiere que Gemma se desempeña mejor acertando a las categorías de manera general, como se ve en la tabla de valores particulares para cada clase, mientras que Nemo logra captar los nombres desconocidos, nótese al observar el Recall de las clases. Gemma tiene dificultades con los nombres ambiguos. No obstante, estos son los elementos que componen una menor parte de la muestra.

Llama y Small tienen peores resultados en precisión y alto recall para mujer, esto indica que catalogan nombres erróneamente como mujer cuando corresponden a otras categorías.

Para efectos de este trabajo, la precisión es la métrica de mayor importancia, puesto que perder un nombre claro es más costoso que acertar ambigüedades con éxito. Esto porque las categorías de desconocido_sigla y desconocido_ambiguo tienen menor presencia en proporción dentro del conjunto total de nombres.

Comparando con el año anterior, los resultados son similares al comparar el subindicador con su metodología del año 2025 y esta nueva versión.
