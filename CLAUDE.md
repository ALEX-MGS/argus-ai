# Argus — contexto para Claude Code

Sistema RAG construido desde cero como plataforma de aprendizaje sobre
arquitectura de sistemas de IA. **No es un producto.** El objetivo es entender
los mecanismos por dentro y aprender a medirlos, no competir con asistentes
existentes.

Léase junto con `README.md` (estado público) y `docs/ARGUS-plan.md`
(diagnóstico técnico y plan por fases).

---

## Estado real

Funciona un pipeline RAG básico de punta a punta: consulta → embedding →
búsqueda FAISS → reordenamiento léxico → top 3 al prompt → respuesta.

La ingestión ya lee archivos del disco y fragmenta por tokens. Lo que sigue
pendiente está en la tabla de limitaciones del `README.md`, y el diagnóstico
detallado de cada punto en `docs/ARGUS-plan.md` §2.

**El bug más importante** es que el umbral de relevancia no filtra:
`IndexFlatL2` devuelve distancia L2 **al cuadrado** (rango 0–4 con vectores
normalizados), así que `threshold=2.0` solo descarta similitud coseno negativa.
En la práctica pasan los 10 resultados siempre, y quien decide qué ve el modelo
termina siendo el reordenamiento léxico, no la similitud semántica.

---

## Comandos

```bash
# Indexar un corpus (cuesta dinero: llama a la API de embeddings)
python -m app.embeddings.index_documents <carpeta> --exclude logs_bench_all_ivf

# Ver cuántos fragmentos y tokens saldrían, sin llamar a la API
python -m app.embeddings.index_documents <carpeta> --dry-run

# Chat por línea de comandos (requiere índice ya construido)
python -m app.main

# Tests: ninguno llama a la API
python -m pytest -q
```

---

## Arquitectura

```
app/
├── core/        config y logging
├── models/      BaseLLM (interfaz) + OpenAILLM (implementación)
├── ingestion/   carga de archivos desde disco
├── processing/  fragmentación por tokens
├── embeddings/  servicio de embeddings, índice FAISS, CLI de indexación
└── main.py      CLI de chat
```

**Regla dura: `app/` es 100 % agnóstico al dominio.** Ni una constante, ni un
prompt, ni un campo específico de ningún caso de uso. Los conjuntos de
evaluación no pueden serlo —medir precisión exige un corpus con respuestas
conocidas— y por eso viven aparte.

---

## Convenciones

- **Español** en docstrings, comentarios, nombres de tests y mensajes de commit.
- Los comentarios explican **por qué**, no qué. Si el código ya dice qué hace,
  el comentario sobra.
- `from __future__ import annotations` y type hints en código nuevo.
- **Ningún test llama a la API.** Si algo necesita red, se sustituye o se salta
  con `pytest.skip` explicando por qué.
- **No se versionan artefactos generados**: `*.index`, `documents.json` y los
  corpus clonados están en `.gitignore`. Se reconstruyen con
  `index_documents.py`.
- Las limitaciones se documentan, no se esconden. Un defecto medido y escrito
  vale más que uno desconocido.

---

## Cómo se decide si un cambio sirve

Con números, no con impresiones.

Cada cambio se mide corriendo el conjunto de evaluación **antes y después**. Si
la métrica no se mueve, el cambio no sirvió. Las métricas que importan:

| Métrica | Qué diagnostica |
|---|---|
| Recall sobre lo **recuperado** | ¿El retriever encontró el documento? |
| Recall sobre lo que llega al **prompt** | ¿El reordenamiento lo conservó? |
| Precisión de respuesta | ¿El modelo usó bien lo que recibió? |
| Tasa de alucinación | ¿Respondió cuando debía abstenerse? |

**Separar los dos recalls es el diagnóstico más útil**: si el primero es alto y
el segundo bajo, el problema está en el reordenamiento y en el corte top-n, no
en los embeddings.

Regla de orden: **medir antes de corregir.** La línea base se toma contra el
sistema sin arreglar, para que cada corrección se compare contra un punto de
partida real y no contra una mejora intuida.

---

## Limitaciones del entorno remoto

Las sesiones de Claude Code en la nube corren tras una política de egress que
**bloquea estos hosts** (403 en el CONNECT):

- `api.openai.com` — no se puede llamar a OpenAI, ni con la API key configurada
- `openaipublic.blob.core.windows.net` — tiktoken no puede descargar su tabla
  de codificación, así que los tests del fragmentador se saltan

**Consecuencia: las corridas reales solo se pueden hacer en la máquina local.**
Desde una sesión remota se puede leer, escribir, diseñar y correr los tests que
usan dobles; no se puede indexar de verdad ni sacar una línea base real.

No intentar rodear la política. Si hace falta acceso, se cambia al crear el
entorno: https://code.claude.com/docs/en/claude-code-on-the-web

---

## Protocolo de trabajo

**Quien tiene teclado, escribe.** Al abrir la sesión, decirlo en una línea.

- Desde el móvil → Claude escribe, el autor revisa
- Desde la máquina local → el autor escribe, Claude revisa

Esto existe porque una vez se implementó la misma refactorización dos veces en
paralelo. No repetirlo.

Otras reglas:

- Una rama por fase, salida de `main`.
- Cambios chicos. Un diff de 500 líneas no se revisa en un teléfono.
- El merge lo hace el autor desde la máquina local, después de correr los evals.
- **El contexto va al repo, no al chat.** Las conclusiones de una conversación
  se commitean como documento. Así cualquier sesión arranca informada sin
  rebriefing.

---

## Inconsistencias conocidas en la documentación

Anotadas para que no confundan a la siguiente sesión:

1. **`README.md` y `docs/ARGUS-plan.md` numeran las fases distinto.** El README
   llama Fase 2 a la ingestión; el plan llama Fase 2 al rediseño de `BaseLLM`.
   Al hablar de fases, nombrar el contenido, no el número.
2. **La tabla de limitaciones del README está desactualizada** en dos puntos: el
   #2 (fragmentación por caracteres) y el #9 (corpus hardcodeado) ya se
   corrigieron. Conviene actualizarla en el próximo cambio que toque el README.

---

## Lo siguiente

El rediseño de `BaseLLM` desbloquea todo lo demás. Su firma actual —un string
entra, un string sale— no admite system prompt separado, mensajes multi-turno,
herramientas ni conteo de tokens. Las fases de recuperación agéntica y
orquestación **no se pueden construir** hasta cambiarla. El diseño ya está
escrito en `docs/ARGUS-plan.md` §2.5; implementar eso y nada más.
