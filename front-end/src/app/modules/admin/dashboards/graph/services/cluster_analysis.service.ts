import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { ClusterAnalysis, ClusterAnalysisRequest, ClusterAnalysisResponse,  } from 'app/core/types/cluster_analysis.type';
import { BehaviorSubject, map, Observable, tap } from 'rxjs';

@Injectable({
    providedIn: 'root'
})
export class ClusterAnalysisService {
    headersRequest = new HttpHeaders({ 'Content-Type': 'application/json' });
    private baseUrl = '/api/v1';

    clusterAnalysis: BehaviorSubject<ClusterAnalysis> = new BehaviorSubject(null);

    constructor(private http: HttpClient) { }

    getClusterAnalysisMetrics(transactionId: number, depth: number = 2): Observable<ClusterAnalysis> {
        const requestObj = new ClusterAnalysisRequest(transactionId, depth);

        const options = { headers: this.headersRequest };

        return this.http.post(`${this.baseUrl}/transductive/cluster-analysis`, requestObj, options)
        .pipe(
            tap((result: ClusterAnalysisResponse) => {
                this.clusterAnalysis.next(result.scores);
            }),
            map((result: ClusterAnalysisResponse) => result.scores)
        )
    }

}
