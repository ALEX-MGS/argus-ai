# Argus

Sistema RAG (Retrieval-Augmented Generation) construido desde cero como plataforma
de aprendizaje sobre arquitectura de sistemas de IA.

El objetivo no es competir con asistentes existentes, sino entender de primera mano
cómo se construye un sistema de recuperación y generación confiable: qué se rompe,
por qué, y cómo se mide.

> **Estado: en desarrollo temprano.** Funciona un pipeline RAG básico end-to-end.
> Las capas de evaluación, monitoreo y orquestación aún no existen.
> Ver [Roadmap](#roadmap).

---

## Qué funciona hoy

Un chat por línea de comandos que responde preguntas sobre un corpus indexado:

1. La consulta se convierte a embedding (`text-embedding-3-small`)
2. Búsqueda de similitud en un índice FAISS local (`IndexFlatL2`)
3. Reordenamiento léxico de los resultados por coincidencia de palabras
4. Los 3 mejores documentos se inyectan como contexto en el prompt
5. El modelo genera la respuesta (OpenAI, configurable por `.env`)
6. Se conserva historial de los últimos 6 turnos de conversación

---

## Estructura actual

```
app/
├── core/
│   ├── config.py              # carga de variables de entorno
│   └── logging_config.py      # configuración de logging (aún sin conectar)
├── models/
│   ├── base_llm.py            # interfaz abstracta de proveedor
│   └── openai_llm.py          # implementación OpenAI
├── ingestion/
│   └── loader.py              # carga de documentos desde el sistema de archivos
├── processing/
│   └── text_splitter.py       # fragmentación medida en tokens
├── retrieval/
│   └── rerank.py              # reordenamiento léxico de los recuperados
├── embeddings/
│   ├── embedding_service.py   # generación de embeddings (individual y por lote)
│   ├── vector_store.py        # índice FAISS + persistencia
│   └── index_documents.py     # CLI de indexación
├── pipeline.py                # consulta → recuperación → respuesta
└── main.py                    # CLI de chat

evals/                         # conjunto de evaluación y harness (ver evals/README.md)
tests/                         # pruebas; no requieren API
```

`pipeline.py` existe para que la CLI y la evaluación entren por el mismo camino.
Si el harness replicara la lógica por su cuenta, mediría una copia en vez del
sistema real.

Todo el código es agnóstico al dominio: no hay lógica específica de ningún caso de uso.

---

## Instalación

```bash
git clone https://github.com/ALEX-MGS/argus-ai.git
cd argus-ai
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Crear un archivo `.env` en la raíz:

```
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=tu_api_key
```

---

## Uso

Construir el índice (requiere llamadas a la API de OpenAI):

```bash
python -m app.embeddings.index_documents
```

Iniciar el chat:

```bash
python -m app.main
```

Escribir `salir` para terminar.

---

## Limitaciones conocidas

Documentadas a propósito: son el trabajo pendiente, no defectos ocultos.

| # | Limitación | Impacto |
|---|-----------|---------|
| 1 | `IndexFlatL2` devuelve distancia L2 **al cuadrado**; con embeddings normalizados el rango es 0–4. El `threshold=2.0` solo descarta similitud coseno negativa. | El filtro de relevancia no filtra casi nada |
| 2 | `text_splitter` corta por caracteres sin respetar límites de palabra ni frase | Fragmenta palabras en documentos reales |
| 3 | El reordenamiento cuenta coincidencias por substring, sin descartar stopwords | Documentos largos e irrelevantes suben de posición |
| 4 | `setup_logging()` está definida pero nunca se invoca | No hay logs en ejecución |
| 5 | Sin conteo de tokens ni seguimiento de costos | No hay visibilidad de gasto |
| 6 | Sin reintentos ni manejo de errores de API | Un rate limit tumba el proceso |
| 7 | `BaseLLM` acepta un string y devuelve un string | No admite system prompts, mensajes multi-turno ni herramientas |
| 8 | Dimensión del índice fija en 1536 | Cambiar de modelo de embeddings rompe el índice en silencio |
| 9 | El corpus está hardcodeado en `index_documents.py` | No hay ingestión de archivos |
| 10 | Sin evaluación automatizada | No hay forma de saber si un cambio mejora o empeora |

---

## Roadmap

Nada de lo siguiente está implementado todavía.

**Fase 1 — Corregir fundamentos**
- Rediseñar `BaseLLM` con interfaz de mensajes (system + multi-turno + herramientas + uso de tokens)
- Migrar a similitud coseno con `IndexFlatIP` y umbral interpretable
- Conectar logging y añadir conteo de tokens y costo
- Reintentos con backoff exponencial

**Fase 2 — Ingestión real**
- Carga de archivos (`.md`, `.txt`, `.pdf`)
- División por tokens respetando límites de frase y párrafo
- Metadatos de origen por fragmento (archivo, posición)

**Fase 3 — Evaluación**
- Conjunto de pares pregunta/respuesta con documento esperado (`must_cite`)
- Métricas separadas de recuperación y de generación
- Casos de ausencia y de trampa para medir alucinación
- Ejecución automatizada antes de cada cambio

**Fase 4 — Recuperación agéntica**
- Búsqueda iterativa: el modelo decide qué consultar y reformula
- Herramientas expuestas por interfaz estándar
- Recuperación híbrida (semántica + léxica)

**Fase 5 — Orquestación**
- Descomposición de tareas en pasos
- Verificación de respuestas contra las fuentes citadas

---

## Dominios de prueba

El código no depende de ningún dominio. Los conjuntos de evaluación sí, porque
medir precisión exige un corpus con respuestas conocidas.

- **Documentación de FAISS** — wiki oficial del proyecto (50 páginas técnicas)
- Dominios adicionales pendientes, para verificar que el núcleo sigue siendo agnóstico

---

## Notas de diseño

**Por qué construirlo desde cero.** Existen frameworks que resuelven esto en menos
líneas. El propósito aquí es entender los mecanismos: por qué una métrica de distancia
mal interpretada invalida un filtro, por qué una estrategia de fragmentación decide la
calidad de la recuperación, por qué sin evaluación no se puede afirmar que algo mejoró.

**Por qué se documentan las limitaciones.** Un sistema cuyos defectos están medidos
y escritos es más confiable que uno cuyos defectos son desconocidos.

---

## Licencia

MIT
