# Argus — Plan de trabajo y notas técnicas

Documento de continuidad. Guardar en la raíz del repo (o en `docs/`).
Contiene el diagnóstico del estado actual, los bugs identificados, el plan por
fases y el diseño del conjunto de evaluación.

---

## 1. Diagnóstico

El proyecto **no está obsoleto en su premisa** — modularidad, evaluación y
observabilidad resultaron ser los ejes correctos. Está **inconcluso en su forma**.

Estado real: 9 archivos de código, 261 líneas. Un pipeline RAG básico funcional.
El README anterior describía once capas; existían dos. Esa brecha entre lo
documentado y lo construido es la causa de la sensación de atraso, no la
tecnología.

**Regla de arquitectura que guía todo el plan:** fijar el camino solo donde se
necesita determinismo (seguridad, cumplimiento, costo, acciones irreversibles).
En todo lo demás, exponer herramientas y dejar que el modelo decida los pasos.
Un pipeline rígido no produce confiabilidad; produce fragilidad ante entradas
que no encajan con la ruta prevista.

---

## 2. Bugs identificados

### 2.1 El umbral de relevancia no filtra (crítico)

`VectorStore` usa `faiss.IndexFlatL2`, que devuelve **distancia L2 al cuadrado**,
no L2. Los embeddings de OpenAI vienen normalizados (norma = 1), así que:

```
L2² = 2 - 2·cos(θ)     →     rango [0, 4]
```

Con `threshold=2.0` solo se descartan documentos con similitud coseno **negativa**.
En un corpus normal eso es casi nada: pasan los 10 resultados siempre.

**Corrección:** migrar a `IndexFlatIP` con vectores normalizados. El producto
interno de vectores unitarios *es* la similitud coseno, con rango [-1, 1]
directamente interpretable. Un umbral de 0.3–0.5 pasa a significar algo real.

Referencia: página *MetricType and distances* del wiki de FAISS.

### 2.2 El splitter nunca se ejercita

`chunk_size=100` está en **caracteres** (~15 palabras). El documento más largo del
corpus actual mide 82 caracteres, así que ningún documento se corta: el módulo
está sin probar. Además corta por índice de string, sin respetar límites de
palabra ni de frase.

**Corrección:** medir en tokens (`tiktoken`, ya está en `requirements.txt` sin
usarse), con `chunk_size` de 300–500 tokens y overlap de ~50, respetando
párrafos y frases.

### 2.3 El rerank premia stopwords

```python
score = sum(1 for word in query.lower().split() if word in text.lower())
```

Dos problemas: en español "de", "la", "el", "que" coinciden con todo, y
`word in text` hace match de **substring** — "de" coincide dentro de "modelo".
Resultado: documentos largos e irrelevantes suben de posición.

**Corrección:** tokenizar por palabra completa, descartar stopwords, y normalizar
por longitud del documento. O sustituir por BM25 y combinarlo con el puntaje
vectorial (recuperación híbrida).

### 2.4 El logging nunca se activa

`setup_logging()` está definida en `core/logging_config.py` y no se invoca en
ningún archivo del repo. El pilar de observabilidad son cuatro líneas muertas.

**Corrección:** llamarla al arranque de `main.py` y registrar por consulta:
query, documentos recuperados con su puntaje, tokens de entrada/salida, costo
estimado y latencia.

### 2.5 `BaseLLM` bloquea todo lo moderno (el más importante)

```python
async def generate(self, prompt: str) -> str
```

Un string entra, un string sale. Esa firma corresponde a la API de *completions*
de 2022–2023. No puede expresar: system prompt separado, mensajes multi-turno,
**tool calls**, structured outputs, streaming, razonamiento extendido, ni uso de
tokens.

Es decir: la abstracción escrita para no depender de un proveedor es lo que
ata el proyecto a una generación de modelos. La recuperación agéntica no cabe
por esta interfaz — por eso el pipeline tiene que ser fijo.

**Corrección (desbloquea las fases 4 y 5):**

```python
@dataclass
class Message:
    role: str                      # "system" | "user" | "assistant" | "tool"
    content: str | list
    tool_calls: list | None = None

@dataclass
class LLMResponse:
    text: str
    tool_calls: list
    input_tokens: int
    output_tokens: int
    stop_reason: str

class BaseLLM(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse: ...
```

### 2.6 Otros

| Problema | Corrección |
|---|---|
| `index_documents.py` ejecuta `asyncio.run(main())` a nivel de módulo | Envolver en `if __name__ == "__main__":` |
| Embeddings pedidos uno por uno en loop | La API acepta arrays: 1 llamada en vez de N |
| Sin reintentos ni manejo de errores | Backoff exponencial ante rate limits y timeouts |
| `dimension=1536` hardcodeada | Derivarla del modelo; validar al cargar el índice |
| Historial recortado con `[-6:]` | Recortar por presupuesto de tokens, no por turnos |
| Prompt armado como un solo string `user` | System prompt separado + array de mensajes |
| `documets.json`, `texts.json`, `faiss.index` versionados | Borrar del repo y añadir a `.gitignore` |
| Corpus hardcodeado en el script | Ingestión desde archivos |

---

## 3. Plan por fases

Estimación con sesiones de 2–3 h, con revisión de cada cambio.
**Total: 11–16 sesiones (~4–6 semanas a 3 sesiones/semana).**

| Fase | Contenido | Sesiones |
|---|---|---|
| 0 | README honesto | 1 |
| 1 | Evals de línea base sobre el sistema actual | 3–5 |
| 2 | `BaseLLM` nuevo + coseno + logging + tokens + reintentos | 2–3 |
| 3 | Ingestión real (archivos, splitter por tokens, metadatos) | 2–3 |
| 4 | Recuperación agéntica (búsqueda iterativa, herramientas, híbrida) | 3–4 |
| 5 | Orquestación (descomposición, verificación contra fuentes) | — |

**Nota sobre el orden:** conviene medir *antes* de corregir. Correr los evals
contra el sistema actual da una línea base numérica, y así cada corrección
posterior queda medida contra un punto de partida en vez de ser una mejora
intuida.

---

## 4. Separación código / dominio

El código en `app/` debe ser **100 % agnóstico**: ni una constante, ni un prompt,
ni un campo específico de ningún dominio.

Los evals **no pueden serlo**: medir precisión exige un corpus con respuestas
conocidas. Un eval genérico no es un eval, es un test unitario.

```
app/            ← agnóstico al dominio
evals/
  ├── faiss_docs/      ← corpus + pares de evaluación
  └── <dominio_2>/     ← segundo dominio: verifica que el core siga agnóstico
```

---

## 5. Corpus de prueba: wiki de FAISS

50 páginas de documentación técnica real, ~395 KB de markdown.

```bash
git clone --depth 1 https://github.com/facebookresearch/faiss.wiki.git
```

Páginas útiles para evaluación: `MetricType-and-distances`, `Faiss-indexes`,
`Guidelines-to-choose-an-index`, `The-index-factory`, `FAQ`,
`Getting-started`, `Faster-search`, `Lower-memory-footprint`.

Excluir del corpus los archivos bajo `logs_bench_all_ivf/` — son volcados de
benchmark de 100–450 KB, sin valor para preguntas y que distorsionan la
recuperación.

Ventaja adicional: es documentación de la librería que el propio proyecto usa,
así que los evals refuerzan conocimiento aplicable directamente.

---

## 6. Diseño del conjunto de evaluación

Formato JSONL, un par por línea. El campo `must_cite` permite medir
**recuperación** y **generación** por separado — sin eso no se puede saber si un
fallo vino del retriever o del modelo.

```jsonl
{"id":"q01","q":"¿Qué devuelve IndexFlatL2 como distancia?","expected":"La distancia L2 al cuadrado, no la L2","must_cite":"MetricType-and-distances.md","type":"factual"}
{"id":"q02","q":"¿Qué índice usar para búsqueda exacta sin compresión?","expected":"IndexFlatL2 o IndexFlatIP","must_cite":"Guidelines-to-choose-an-index.md","type":"factual"}
{"id":"q03","q":"¿Por qué IndexFlatIP con vectores normalizados equivale a similitud coseno?","expected":"Porque el producto interno de dos vectores unitarios es igual al coseno del ángulo entre ellos","must_cite":"MetricType-and-distances.md","type":"explicativa"}
{"id":"q04","q":"Si uso IVF y quiero más precisión a costa de velocidad, ¿qué parámetro ajusto y en qué dirección?","expected":"Aumentar nprobe","must_cite":["Faiss-indexes.md","Faster-search.md"],"type":"multi_hop"}
{"id":"q05","q":"¿Qué garantías de precisión da FAISS con IVF sobre la búsqueda exhaustiva?","expected":"NO_SE / la documentación no especifica garantías formales","must_cite":null,"type":"ausencia"}
{"id":"q06","q":"¿Cómo se configura la replicación distribuida de FAISS en Kubernetes?","expected":"NO_SE","must_cite":null,"type":"trampa"}
```

### Tipos y propósito

| Tipo | Qué mide |
|---|---|
| `factual` | Recuperación básica: dato único y literal |
| `multi_hop` | Necesita 2+ documentos a la vez; revienta el `top_docs[:3]` actual |
| `ausencia` | La respuesta correcta es "no está en el corpus" — donde más se alucina |
| `trampa` | Algo que nunca estuvo en el corpus; detecta alucinación pura |

### Distribución objetivo (~30 pares)

15 factual · 8 multi_hop · 5 ausencia · 2 trampa

El sesgo hacia los casos difíciles es intencional. Si todos son factuales, el
eval marca 95 % desde el primer día y no enseña nada.

### Calificación

- **`expected` corto:** comparación por substring o regex.
- **Respuestas explicativas:** LLM como juez con rúbrica **binaria**
  (*¿contiene el hecho X? sí/no*). Nunca pedir una nota del 1 al 10 — es ruido.
- **`must_cite`:** se evalúa aparte, comparando contra los chunks que el
  retriever efectivamente devolvió.

### Métricas a reportar

| Métrica | Definición |
|---|---|
| Recall@k | ¿El chunk de `must_cite` estuvo entre los k recuperados? |
| Precisión de respuesta | % de respuestas que contienen el hecho esperado |
| Tasa de alucinación | % de `ausencia` + `trampa` respondidas con algo distinto a "no sé" |
| Tokens / costo por consulta | Presupuesto por pregunta |

---

## 7. Skill vs. sistema propio

Distinción para no poner las dos cosas a competir:

- **Una skill** son instrucciones que un asistente lee cuando aplican; se apoya en
  su razonamiento y sus herramientas. Rápida de escribir, funciona hoy. Límite:
  vive dentro del asistente y requiere una conversación abierta.
- **Argus** es un sistema propio: corre solo, procesa volumen sin supervisión, se
  conecta a lo que sea, no depende de un tercero. Cuesta semanas y al principio
  rinde peor.

Para resolver una tarea concreta hoy, la skill gana. Argus se justifica por otras
tres razones: entender el mecanismo (no es lo mismo que usarlo), construir
evidencia empleable (evals, context engineering, MCP), y cubrir casos donde no
puede haber un asistente en el loop.

No tratarlo como reemplazo de una skill — perdería siempre. Tratarlo como
escuela de arquitectura de sistemas.

---

## 8. Próximo paso inmediato

1. Reemplazar el README (archivo aparte, ya generado).
2. Limpiar el repo: borrar `documets.json`, `texts.json`, `faiss.index`; actualizar `.gitignore`.
3. Clonar el wiki de FAISS y filtrar `logs_bench_all_ivf/`.
4. Escribir los ~30 pares de evaluación.
5. Correr la línea base contra el sistema **sin corregir** y anotar los números.
