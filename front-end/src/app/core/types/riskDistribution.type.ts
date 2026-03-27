export interface RiskDistribution {
    series: number[];
    labels: string[];
}

export class RiskDistributionModel implements RiskDistribution {
    constructor(
        public series: number[],
        public labels: string[]
    ) { }
}