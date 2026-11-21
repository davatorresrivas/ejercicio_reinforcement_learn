# Optimización de una mesa de ayuda con Reinforcement Learning

Proyecto académico: comparación de **Q-Learning**, **SARSA** y una **política fija** para asignar agentes de soporte técnico.

## Problema

Una mesa de ayuda debe decidir cuántos agentes dedicar a solicitudes de soporte en intervalos de 30 minutos. Se busca equilibrar el costo de personal y los tickets pendientes. **No se modela un SLA real ni la complejidad de cada ticket.** Los costos son unidades ficticias.

## Modelo

- 5 agentes disponibles; se inicia con 3 asignados.
- 8 intervalos de 30 minutos por jornada.
- Cada agente atiende hasta 3 tickets por intervalo.
- Demanda Poisson: niveles `[0, 0, 1, 2, 2, 1, 1, 0]` y medias `[3, 7, 12]`.
- Estado observable: `(intervalo, nivel de demanda, pendientes, agentes)`.
- Acciones: `-1`, `0`, `+1`, sujetas a límites de personal.
- Recompensa: `-(10 * agentes + 5 * pendientes)` por intervalo; al cierre se restan otras `10 * pendientes`.
- Política fija: 3 agentes toda la jornada.

Las llegadas se generan de antemano para poder evaluar políticas sobre las mismas jornadas. La observación del agente **no incluye llegadas futuras**.

## Estructura

```text
helpdesk-rl/
├── notebooks/mesa_ayuda_reinforcement_learning.ipynb
├── src/environment.py
├── src/agents.py
├── src/evaluation.py
├── tests/test_environment.py
├── main.py
├── requirements.txt
└── results/
    ├── figures/
    └── metrics/
```

## Instalación

Requiere Python 3.10 o superior. Desde la carpeta raíz:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar

**Notebook:** `jupyter notebook notebooks/mesa_ayuda_reinforcement_learning.ipynb` y ejecutar todas las celdas en orden. También se puede abrir en Google Colab.

**Scripts:**

```bash
python main.py
```

Esto entrena cada algoritmo durante 10.000 episodios, evalúa 1.000 jornadas compartidas y ejecuta una prueba de robustez con cinco semillas de entrenamiento. Guarda CSV en `results/metrics/` y PNG en `results/figures/`.

Para una comprobación rápida:

```bash
python main.py --episodes 100 --test-days 20 --skip-robustness
```

**Pruebas:**

```bash
python -m unittest discover -s tests -v
```

## Reproducibilidad y análisis

Las semillas principales de entrenamiento son `42`, las jornadas de prueba `10000..10999` y la robustez usa semillas de entrenamiento `11,22,33,44,55` con jornadas `20000..20199`. Las pruebas son independientes de las jornadas de entrenamiento. Las métricas incluyen recompensa, costo de personal, costo por cola, penalización terminal, pendientes y proporción atendida.

**Interpretación:** una recompensa mayor (menos negativa) significa menor costo penalizado, no necesariamente mejor servicio en un sistema real. No se afirma convergencia formal ni generalización a mesas de ayuda reales.

## Integrantes y aportes

Completar antes de entregar:

| Integrante | Aporte |
|---|---|
| [Nombre 1] | [Aporte] |
| [Nombre 2] | [Aporte] |
| [Nombre 3] | [Aporte] |

## Informe técnico

Adjuntar el informe PDF por separado o en una carpeta `docs/`, incluyendo el enlace público al repositorio GitHub, resultados reales y referencias APA.
