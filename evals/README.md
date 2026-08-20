# Evaluación

Mide el pipeline RAG contra un corpus con respuestas conocidas. Sin esto, cada
cambio al sistema es una mejora intuida en vez de una mejora medida.

## Por qué el corpus no está en el repo

El corpus se clona aparte y está en `.gitignore`. Versionarlo mezclaría 21 MB de
documentación ajena con el código propio. Para obtenerlo:

```bash
git clone --depth 1 https://github.com/facebookresearch/faiss.wiki.git \
    evals/faiss_docs/corpus
rm -rf evals/faiss_docs/corpus/.git
```

Son 50 archivos `.md` (~492 KB) más una carpeta `logs_bench_all_ivf/` con
volcados de benchmark que hay que excluir al indexar: no responden preguntas y
distorsionan la recuperación.

Se eligió el wiki de FAISS porque es documentación técnica real y es la librería
que el propio proyecto usa, así que evaluar sobre ella refuerza conocimiento
aplicable.

## Cómo correrlo

```bash
# 1. Verificación de la tubería: sin API, sin costo, resultados deterministas
python -m evals.run --stub

# 2. Corrida real
python -m app.embeddings.index_documents evals/faiss_docs/corpus \
    --exclude logs_bench_all_ivf --dry-run     # revisa el costo primero
python -m app.embeddings.index_documents evals/faiss_docs/corpus \
    --exclude logs_bench_all_ivf
python -m evals.run
```

El reporte queda en `evals/results/`. Los parámetros del pipeline se pueden
barrer sin tocar código:

```bash
python -m evals.run --top-n 5 --threshold none
```

## El conjunto de preguntas

`faiss_docs/qa.jsonl`, un caso por línea:

```json
{"id": "f01", "type": "factual", "q": "...", "expected": "...",
 "expected_any": ["al cuadrado"], "must_cite": "MetricType-and-distances.md"}
```

| Campo | Para qué |
|---|---|
| `type` | Determina cómo se puntúa el caso |
| `expected` | Respuesta correcta, legible para humanos |
| `expected_any` | Si **alguna** de estas cadenas aparece, cuenta como acierto |
| `must_cite` | Archivos que deberían sustentar la respuesta; mide recall |

`must_cite` es lo que permite separar un fallo del retriever de un fallo del
modelo. Sin ese campo, una respuesta mala no dice cuál de los dos falló.

### Distribución

| Tipo | N | Qué mide |
|---|---|---|
| `factual` | 15 | Recuperación básica: un dato único y literal |
| `multi_hop` | 8 | Exige 2+ documentos a la vez; presiona el corte `top_n` |
| `ausencia` | 5 | El corpus toca el tema pero no da el dato: hay que abstenerse |
| `trampa` | 2 | Nunca estuvo en el corpus: detecta alucinación pura |

El sesgo hacia los casos difíciles es deliberado. Un conjunto solo de preguntas
factuales marca 95 % el primer día y no enseña nada.

Cada `ausencia` se verificó contra el corpus antes de etiquetarla. Una etiqueta
`NO_SE` puesta sobre algo que sí está documentado invierte la métrica: premiaría
al modelo justo por no encontrar lo que sí estaba.

`a04` merece mención aparte: pregunta por el valor por defecto de `nlist`. El
tutorial del corpus usa `nlist = 100` como ejemplo, no como default, así que el
contexto recuperado contiene un número muy tentador que **no** responde la
pregunta. Es la clase de caso donde se ve si el modelo distingue entre «lo vi en
el contexto» y «el contexto lo afirma».

## Métricas

| Métrica | Definición |
|---|---|
| `recall_retrieved` | ¿El archivo de `must_cite` estuvo entre los *k* recuperados? |
| `recall_context` | ¿Y entre los que efectivamente entraron al prompt? |
| `accuracy` | Fracción de respuestas que contienen el hecho esperado |
| `hallucination_rate` | De los casos que exigían abstenerse, cuántos respondieron igual |
| `avg_prompt_tokens` | Presupuesto de entrada por consulta |

**La diferencia entre los dos recalls es el diagnóstico más útil del reporte.**
Si `recall_retrieved` es alto y `recall_context` es bajo, el retriever encontró
el documento y el rerank lo descartó: el problema está en el reordenamiento y en
el corte `top_n`, no en los embeddings.

`avg_output_tokens` sale como `n/d`: `BaseLLM.generate` devuelve un string y
descarta el objeto de uso de la API. Se resuelve al rediseñar la interfaz
(`docs/ARGUS-plan.md` §2.5).

## Sobre el modo `--stub`

Sustituye embeddings y modelo por dobles deterministas. Sirve para comprobar que
la cadena completa funciona sin gastar en API, y para que la primera corrida real
no se vaya en errores de plomería.

**No mide calidad.** Los embeddings son bolsa de palabras con hashing, así que la
recuperación es léxica y no semántica, y el modelo no razona: devuelve el
contexto que recibió. Una corrida con dobles solo demuestra que las piezas
encajan.

## Orden de trabajo

Medir antes de corregir. La primera corrida real debe hacerse contra el sistema
**sin arreglar**, incluido el umbral roto de `docs/ARGUS-plan.md` §2.1. Ese
reporte es la línea base; cada corrección posterior se compara contra él en vez
de ser una mejora intuida.
