import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { MLModelType } from './model_type.enum';
import { map, Observable } from 'rxjs';
import { EvaluationMetrics } from './evaluation_metrics.type';
import { LossResultsResponse } from './loss_results.type';

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
}