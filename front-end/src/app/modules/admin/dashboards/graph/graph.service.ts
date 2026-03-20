import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { HistoricScoreResponse, HistoryScore } from 'app/core/types/history_score.type';
import { BehaviorSubject, map, Observable, tap } from 'rxjs';

@Injectable({
    providedIn: 'root'
})
export class GraphDashboardService {
    headersRequest = new HttpHeaders({ 'Content-Type': 'application/json' });
    private baseUrl = '/api/v1';

    selectedScore: BehaviorSubject<HistoryScore> = new BehaviorSubject(null);
    historyScores: BehaviorSubject<HistoryScore[]> = new BehaviorSubject([]);

    constructor(private http: HttpClient) { }

    getHistoricScoresMetrics(top_list: number = 5): Observable<HistoryScore[]> {
        return this.http.get(`${this.baseUrl}/transductive/history/${top_list}`)
        .pipe(
            tap((result: HistoricScoreResponse) => {
                this.historyScores.next(result.scores);
            }),
            map((result: HistoricScoreResponse) => result.scores)
        )
    }

    setSelectedScore(score: HistoryScore) {
        this.selectedScore.next(score);
    }
}
