from fastapi import APIRouter

from back_end.src.app.services.transductive_service import TransductiveScoringService
from back_end.src.app.services.inductive_service import InductiveScoringService
from back_end.src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel
from back_end.src.app.schemas.loss_results import LossResultsResponse
from back_end.src.app.schemas.loss_results import LossResults

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
transductive_service = TransductiveScoringService(model)
inductive_service = InductiveScoringService(model)

@router.get("/transductive/loss-results", response_model=LossResultsResponse)
def transductive_loss_results():

    loss_results = transductive_service.get_model_train_validation_results()

    if loss_results is not None:
        train_results = [LossResults(epoch=item["name"], value=item["value"]) for item in loss_results["train_results"]]
        val_results = [LossResults(epoch=item["name"], value=item["value"]) for item in loss_results["val_results"]]
        
        return LossResultsResponse(train_loss = train_results, val_loss = val_results)

    else:
        return LossResultsResponse(train_loss = None, val_loss = None)

@router.get("/inductive/loss-results", response_model=LossResultsResponse)
def inductive_evaluation_metrics():
    loss_results = inductive_service.get_model_train_validation_results()

    if loss_results is not None:

        train_results = [LossResults(epoch=item["name"], value=item["value"]) for item in loss_results["train_results"]]
        val_results = [LossResults(epoch=item["name"], value=item["value"]) for item in loss_results["val_results"]]

        return LossResultsResponse(train_loss = train_results, val_loss = val_results)
    
    else:
        return LossResultsResponse(train_loss = None, val_loss = None)
