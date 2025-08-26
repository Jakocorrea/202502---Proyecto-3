# Paint Booth Ventilation Simulation

Implementación en dos archivos:
- `sim_logic.py`: **lógica** del modelo (hidráulica, emisiones, concentraciones, energía).
- `run_sim.py`: **consola** (lee JSON, ejecuta la simulación, exporta CSV/JSON y, opcionalmente, PNG).

## Requisitos
- Python 3.9+  
- (Opcional) `matplotlib` si quieres gráficos PNG con `--plots 1`.

## Ejecución rápida
1. Ajusta `example_config.json` con tus datos.
2. Corre:
   ```bash
   python run_sim.py --config example_config.json --outdir outputs --dt 1 --tend auto --plots 0
