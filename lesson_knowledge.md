# Conocimiento Teórico: Construcción de Agentes con Herramientas

Esta lección cubre la transición de modelos de lenguaje estáticos a **Agentes Inteligentes** capaces de interactuar con el mundo real.

## 1. Tool-Calling (Llamada a Herramientas)
Los LLMs modernos (como `qwen2.5`, `gpt-4`, etc.) han sido entrenados para reconocer cuándo una pregunta requiere una función externa.
-   **Esquema JSON**: Se le pasa al modelo una descripción de la función (nombre, parámetros, tipos).
-   **Decisión**: El modelo no *ejecuta* la función, sino que devuelve un mensaje con el nombre de la herramienta y los argumentos que quiere usar.

## 2. El Patrón ReAct (Reasoning + Acting)
Es el "cerebro" detrás de un agente. Sigue un ciclo iterativo:
1.  **Reasoning (Razonamiento)**: El modelo piensa qué paso dar a continuación.
2.  **Acting (Acción)**: El modelo decide llamar a una herramienta.
3.  **Observation (Observación)**: El sistema ejecuta la herramienta y le devuelve el resultado al modelo.
4.  **Repetición**: El modelo analiza el resultado y decide si necesita más herramientas o si ya puede dar la respuesta final.

## 3. Implementación Manual vs. LangGraph

### Implementación Manual
Requiere gestionar manualmente:
-   El historial de mensajes (`HumanMessage`, `AIMessage`, `ToolMessage`).
-   El bucle `for` o `while` de ejecución.
-   El mapeo de nombres de herramientas a funciones ejecutables.
-   El manejo de IDs de llamadas a herramientas (`tool_call_id`).

### LangGraph (`create_react_agent`)
Es una abstracción de alto nivel que:
-   **Automatiza el bucle ReAct**: No necesitas escribir ciclos `for`.
-   **Gestión de Estado**: Mantiene el historial de mensajes automáticamente.
-   **Grafos de Control**: Permite crear flujos mucho más complejos donde el agente puede "bifurcarse" o volver atrás basándose en condiciones.

## 4. Conceptos Clave de LangChain
-   **`bind_tools`**: Vincula las definiciones de las herramientas al modelo.
-   **`invoke`**: Envía el estado actual al modelo o al agente.
-   **`ToolMessage`**: Mensaje específico para enviar el resultado de una herramienta de vuelta al modelo. Es crucial para que el modelo sepa "qué pasó" después de su petición.
