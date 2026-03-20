export interface KpiSummary {
    total: number;
    highRisk: number;
    mediumRisk: number;
    lowRisk: number;
    meanFraudProbability: number;
}


export class KpiSummaryModel implements KpiSummary {
    constructor(
        public total: number,
        public highRisk: number,
        public mediumRisk: number,
        public lowRisk: number,
        public highRiskPercent: number,
        public mediumRiskPercent: number,
        public lowRiskPercent: number,
        public meanFraudProbability: number
    ) { }

    static fromTransactions(transactions: any[]): KpiSummaryModel {
        const total = transactions.length;
        const highRisk = transactions.filter(t => t.risk_level === 'high' || t.risk_level === 'critical').length;
        const mediumRisk = transactions.filter(t => t.risk_level === 'medium').length;
        const lowRisk = transactions.filter(t => t.risk_level === 'low').length;
        const mean = transactions.reduce((sum, t) => sum + t.fraud_probability, 0) / total;

        const highRiskPercent = highRisk / total;
        const mediumRiskPercent = mediumRisk / total;
        const lowRiskPercent = lowRisk / total;


        return new KpiSummaryModel(
            total,
            highRisk,
            mediumRisk,
            lowRisk,
            highRiskPercent,
            mediumRiskPercent,
            lowRiskPercent,
            mean);
    }
}