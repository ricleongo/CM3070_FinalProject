import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { MLModelType } from './model_type.enum';
import { map, Observable } from 'rxjs';
import { EvaluationMetrics } from './evaluation_metrics.type';
import { LossResultsResponse } from './loss_results.type';
import { MetricsType } from './metrics_type.enum';

@Injectable({
    providedIn: 'root'
})
export class AnalysisService {
    headersRequest = new HttpHeaders({ 'Content-Type': 'application/json' });
    private baseUrl = '/api/v1';

    constructor(private http: HttpClient) { }

    getEvaluationMetrics(modelType: MLModelType): Observable<EvaluationMetrics> {
        const model = MLModelType[modelType].toLowerCase();

        return this.http.get(`${this.baseUrl}/${model}/evaluation-metrics`)
            .pipe(
                map((data: { metrics: EvaluationMetrics }) => {

                    return data.metrics;
                })
            );
    }

    getLossResults(modelType: MLModelType): Observable<LossResultsResponse> {
        const model = MLModelType[modelType].toLowerCase();

        return this.http.get(`${this.baseUrl}/${model}/loss-results`)
            .pipe(
                map((loss_results: LossResultsResponse) => {
                    return loss_results
                })
            );
    }

    // confusion-matrix.component.ts

    getBenchmark(metricsType: MetricsType, value: number): {
        label: string; color: string; description: string
    } {
        const benchmarks = {
            0: [
                { max: 0.50, label: 'Poor', color: '#dc2626', description: 'Below rule-based AML systems' },
                { max: 0.70, label: 'Acceptable', color: '#d97706', description: 'Typical legacy bank AML' },
                { max: 0.85, label: 'Good', color: '#16a34a', description: 'Modern ML-based AML range' },
                { max: 1.00, label: 'Excellent', color: '#0d9488', description: 'Best-in-class crypto AML' },
            ],
            1: [
                { max: 0.10, label: 'Excellent', color: '#0d9488', description: 'Best-in-class commercial tools' },
                { max: 0.30, label: 'Good', color: '#16a34a', description: 'Acceptable for pre-screening' },
                { max: 0.60, label: 'Acceptable', color: '#d97706', description: 'Requires heavy analyst triage' },
                { max: 1.00, label: 'Poor', color: '#dc2626', description: 'Unsustainable — alert fatigue' },
            ],
            2: [
                { max: 0.02, label: 'Low', color: '#dc2626', description: 'Missing significant activity' },
                { max: 0.05, label: 'Targeted', color: '#16a34a', description: 'Ideal for ~2% illicit networks' },
                { max: 0.15, label: 'Broad Sweep', color: '#d97706', description: 'High recall, collateral flags' },
                { max: 1.00, label: 'Excessive', color: '#dc2626', description: 'Over-reporting SAR risk' },
            ],
        };

        return benchmarks[metricsType].find(b => value <= b.max)!;
    }

    // getMetricsBenchmarks(metricsType: MetricsType) {
    //     const FDR_BENCHMARKS = {
    //         poor: { max: 0.50, label: 'Poor', color: '#dc2626' },
    //         acceptable: { max: 0.70, label: 'Acceptable', color: '#d97706' },
    //         good: { max: 0.85, label: 'Good', color: '#16a34a' },
    //         excellent: { max: 1.00, label: 'Excellent', color: '#0d9488' },
    //     };

    //     const FAR_BENCHMARKS = {
    //         excellent: { max: 0.10, label: 'Excellent', color: '#0d9488' },
    //         good: { max: 0.30, label: 'Good', color: '#16a34a' },
    //         acceptable: { max: 0.60, label: 'Acceptable', color: '#d97706' },
    //         poor: { max: 1.00, label: 'Poor', color: '#dc2626' },
    //     };

    //     const NRC_BENCHMARKS = {
    //         low: { max: 0.02, label: 'Low Coverage', color: '#dc2626' },
    //         targeted: { max: 0.05, label: 'Targeted', color: '#16a34a' },
    //         broad: { max: 0.15, label: 'Broad Sweep', color: '#d97706' },
    //         excessive: { max: 1.00, label: 'Excessive Flagging', color: '#dc2626' },
    //     };

    //     switch (metricsType) {
    //         case MetricsType.FDR:
    //             return FDR_BENCHMARKS;
    //         case MetricsType.FAR:
    //             return FAR_BENCHMARKS;
    //         case MetricsType.NRC:
    //             return NRC_BENCHMARKS;
    //         default:
    //             return null;
    //     }

    // }

}