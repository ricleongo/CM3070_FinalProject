export interface ClusterAnalysis {
    transaction_id: number;
    cluster_size: number;
    cluster_risk_mean: number;
    cluster_risk_max: number;
    suspicious_nodes: number;
}

export class ClusterAnalysisModel implements ClusterAnalysis {
    constructor(
        public transaction_id: number,
        public cluster_size: number,
        public cluster_risk_mean: number,
        public cluster_risk_max: number,
        public suspicious_nodes: number
    ) { }
}


export class ClusterAnalysisRequest {
    constructor(
        public transaction_id: number,
        public hop_depth: number
    ) { }
}

export class ClusterAnalysisResponse {
    public scores: ClusterAnalysis;
}
