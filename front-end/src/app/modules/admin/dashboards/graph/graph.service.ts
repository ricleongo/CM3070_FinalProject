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
    historyScores: BehaviorSubject<HistoryScore[]> = new BehaviorSubject(null);

    constructor(private http: HttpClient) { }

    getHistoricScoresMetrics(top_list: number = 5): Observable<HistoryScore[]> {
        // Create a mini cache system.
        const history = this.historyScores?.value?.length > 0? this.historyScores.asObservable() : null;

        return history?? this.http.get(`${this.baseUrl}/transductive/history/${top_list}`)
        .pipe(
            map((result: HistoricScoreResponse) => {

                this.historyScores.next(result.scores);

                return result.scores;
            })
        )
    }

    setSelectedScore(score: HistoryScore) {
        this.selectedScore.next(score);
    }
}
