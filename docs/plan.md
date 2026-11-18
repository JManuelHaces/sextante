# Librería de interpretabilidad para la tesis — plan y nombre

Nov 18, 2026 · @José Manuel Haces

## Nombre

**Elegido: sextante** (18 nov 2026). Libre en PyPI. En GitHub solo coincide con SEXTANTE, el antiguo framework de geoprocesamiento de gvSIG/QGIS (otro dominio).

| Nombre | Línea | Comentario |
| --- | --- | --- |
| trasluz | Médica / transparencia | Favorito. Corto, original, en español y fácil de pronunciar en inglés |
| contraluz | Médica / transparencia | Variante de la misma idea |
| diafano | Médica / transparencia | Variante; sin acento por PyPI |
| sextante | Náutica (sigue a Timón) | Instrumento para orientarse midiendo ángulos; encaja con "medir" el modelo |
| astrolabio | Náutica (sigue a Timón) | Más poético, pero largo para cada import |
| latenscope | Descriptivo en inglés | Se entiende, menos memorable |
| vlm-lens | Descriptivo en inglés | Se entiende, menos memorable |
| medsae | Descriptivo en inglés | Se entiende, menos memorable |

- [x] Verificar que el nombre elegido esté libre en GitHub y que no choque con un proyecto conocido (solo se revisó PyPI)

## Plan de la librería

Una librería pública y genérica en PyTorch puro, probada primero con 1 o 2 modalidades. Se construye con Claude Code empezando por la extracción de activaciones, porque todo lo demás depende de ella.

### Principios

- PyTorch puro con hooks propios, sin TransformerLens ni nnsight.
- Genérica para cualquier VLM de HuggingFace. Cada arquitectura nueva entra con un adaptador pequeño que indica dónde están la torre de visión, el proyector y las capas del LLM.
- Se prueba con 1 o 2 modalidades (por ejemplo, tórax y fondo de ojo).
- Desarrollo local con modelos diminutos inicializados al azar para tests rápidos. Las corridas reales van en una GPU rentada.

### Modelos sugeridos

- **Principal: MedGemma 4B (multimodal).** Está basado en Gemma 3, así que aprovecha la experiencia previa con GemmaScope y SAELens. Podría compararse contra SAEs públicos de Gemma 3, si existen para esa versión.
- **Segundo, para transferencia entre arquitecturas: LLaVA-Med.** Tiene encoder y LLM distintos, lo que prueba que la librería de verdad es genérica.

### Módulos

| Módulo | Qué hace |
| --- | --- |
| `adapters` | Descripción de cada arquitectura (MedGemma, LLaVA-Med) y registro para añadir más |
| `activations` | Hooks, extracción por lotes y guardado en disco (safetensors o memmap) con metadatos de imagen, reporte, idioma y capa |
| `sae` | Entrenamiento (TopK o JumpReLU), carga y guardado, métricas de reconstrucción y de features muertas |
| `features` | Ejemplos que más activan cada feature, heatmaps sobre parches de imagen, tokens del reporte, autointerp con LLM y estadísticas de propiedad del modelo vs. artefacto del corpus |
| `validation` | Exportar paquetes de features para que los anoten los clínicos, importar anotaciones y calcular acuerdo (kappa, etc.) |
| `intervene` | Steering y ablation durante la generación de reportes, con métricas de alucinación antes y después |
| `transfer` | Comparar features entre EN/ES, entre arquitecturas y entre modalidades |
| Base transversal | Configs, semillas, caché y logging de experimentos |

### Orden de construcción

1. Esqueleto del repo: `pyproject` con layout `src/`, tests y CI.
2. `adapters` y `activations`, con un modelo diminuto en tests y MedGemma en la GPU.
3. `sae`, después `features`.
4. `validation`, a tiempo para cuando se sumen los especialistas.
5. `intervene` y `transfer`.
6. Documentación y release público.
