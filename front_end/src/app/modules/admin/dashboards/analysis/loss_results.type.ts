export interface LossResults {
    epoch: string,
    value: number
}

export interface LossResultsResponse {
    train_loss: LossResults[],
    val_loss: LossResults[]
}
