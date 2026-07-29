from settings.settings import settings
from modelos.dinamico.identificacion import identify
from modelos.dinamico.controlador import default_cost_matrices, calculate_lqr_gain
from utils.file_helpers import load_baseline_model, load_json_to_dict
from modelos.evaulacion.strategies import LogisticSimulationStrategy, DynamicSimulationStrategy


class SimulatorFactory:
    @staticmethod
    def create_logistic() -> LogisticSimulationStrategy:
        model = load_baseline_model(settings.LOGISTICS_MODEL_PATH)

        return LogisticSimulationStrategy(model)

    @staticmethod
    def create_dynamic() -> DynamicSimulationStrategy:
        A, B, C = identify()
        Q, R = default_cost_matrices()
        K = calculate_lqr_gain(A, B, Q, R)
        scale_params = load_json_to_dict(settings.MATRIX_SYSTEM_SCALE_PATH)

        return DynamicSimulationStrategy(
            A,
            B,
            C,
            K,
            Q,
            R,
            scale_params
        )
        
