export interface HistoryScore {
    transaction_id: number,
    fraud_probability: number,
    risk_level: 'high' | 'medium' | 'low'
}

export interface HistoricScoreResponse {
    scores: HistoryScore[]
}
