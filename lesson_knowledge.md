# Conocimiento Teórico: Construcción de Agentes con Herramientas

Esta lección cubre la transición de modelos de lenguaje estáticos a **Agentes Inteligentes** capaces de interactuar con el mundo real.

## 1. Anatomía de una Herramienta (Tool)
Para que un LLM pueda usar una función, esta debe estar estructurada adecuadamente:
-   **Decorador `@tool`**: Registra la función en LangChain y genera automáticamente un esquema JSON de validación.
-   **Nombre de la Función**: El LLM lo usa como identificador. Debe ser descriptivo.
-   **Type Annotations (Anotaciones de Tipo)**: Ayudan al modelo a entender qué tipo de datos debe enviar (ej. `str`, `int`, `List`).
-   **Docstring**: Es el elemento más crítico. Describe **qué** hace la herramienta y **cuándo** usarla. El LLM "lee" este texto para decidir si invoca la herramienta.

## 2. Tool-Calling (Llamada a Herramientas)
Los LLMs modernos han sido entrenados para identificar intenciones.
-   **Decisión**: El modelo devuelve un objeto `tool_calls` con:
    -   `name`: Nombre de la herramienta.
    -   `args`: Argumentos en formato diccionario.
    -   `id`: Identificador único para rastrear la respuesta.
-   **ToolMessage**: Es el mensaje que contiene el resultado de la ejecución. Es obligatorio incluir el `tool_call_id` para vincular la respuesta con la petición original.

## 3. Automatización con LCEL (LangChain Expression Language)
Antes de usar agentes completos, se pueden crear cadenas automatizadas usando:
-   **`RunnablePassthrough`**: Permite mantener el estado (mensajes previos) mientras se añaden nuevos datos (resultados de herramientas).
-   **`RunnableLambda`**: Se usa para transformar la salida final o ejecutar lógica personalizada.
-   **Encadenamiento con `|`**: Permite conectar pasos: `Prompt | LLM | ToolExecution | FinalLLM`.

## 4. El Patrón ReAct (Reasoning + Acting)
Es el "cerebro" detrás de un agente. Sigue un ciclo iterativo:
1.  **Reasoning (Razonamiento)**: El modelo piensa qué paso dar a continuación.
2.  **Acting (Acción)**: El modelo decide llamar a una herramienta.
3.  **Observation (Observación)**: El sistema ejecuta la herramienta y le devuelve el resultado.
4.  **Repetición**: El modelo analiza el resultado y decide si necesita más herramientas.

## 5. Implementación Manual vs. LangGraph
### Implementación Manual (Recursiva)
Requiere crear una función que se llame a sí misma (`should_continue`) hasta que no haya más llamadas a herramientas pendientes. Es útil para entender el flujo, pero complejo de mantener.

### LangGraph (`create_react_agent`)
Es la evolución natural que gestiona todo lo anterior:
-   **Bucle Automático**: Elimina la necesidad de recursión manual.
-   **Persistencia**: Capacidad de guardar el estado del agente (checkpoints).
-   **Flexibilidad**: Permite definir grafos con nodos y bordes para flujos de trabajo no lineales.
