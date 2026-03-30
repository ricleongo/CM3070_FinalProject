from fastapi import APIRouter

from back_end.src.app.services.transductive_service import TransductiveScoringService
from back_end.src.app.services.inductive_service import InductiveScoringService
from back_end.src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel
from back_end.src.app.schemas.evaluation_metrics import EvaluationMetricsResponse, EvaluationMetrics

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
transductive_service = TransductiveScoringService(model)
inductive_service = InductiveScoringService(model)

@router.get("/transductive/evaluation-metrics")
def transductive_evaluation_metrics():
    evaluation_metrics = transductive_service.get_model_evaluation_results()

    if evaluation_metrics is not None:
        loss, auc, precision, recall, f1, fdr, nrc, far = evaluation_metrics.values()

        result = EvaluationMetrics(
            loss = loss,
            auc = auc,
            precision = precision,
            recall = recall,
            f1 = f1,
            fdr = fdr,
            nrc = nrc,
            far = far
        )
        
        return EvaluationMetricsResponse(metrics = result)    

    else:
        return EvaluationMetricsResponse(metrics = None)

@router.get("/inductive/evaluation-metrics", response_model=EvaluationMetricsResponse)
def inductive_evaluation_metrics():
    evaluation_metrics = inductive_service.get_model_evaluation_results()

    if evaluation_metrics is not None:
        loss, auc, precision, recall, f1, fdr, nrc, far = evaluation_metrics.values()

        result = EvaluationMetrics(
            loss = loss,
            auc = auc,
            precision = precision,
            recall = recall,
            f1 = f1,
            fdr = fdr,
            nrc = nrc,
            far = far
        )
        
        return EvaluationMetricsResponse(metrics = result)
    
    else:
        return EvaluationMetricsResponse(metrics = None)
