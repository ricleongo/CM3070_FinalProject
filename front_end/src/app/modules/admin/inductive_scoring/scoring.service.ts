import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { HistoryScore } from 'app/core/types/history_score.type';
import { BehaviorSubject, map, Observable, tap } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ScoringService {
    private _simulated_crypto_data: BehaviorSubject<any> = new BehaviorSubject(null);
    
    headersRequest = new HttpHeaders({ 'Content-Type': 'application/json' });
    private baseUrl = '/api/v1';

    /**
     * Constructor
     */
    constructor(private _httpClient: HttpClient) {}

    // -----------------------------------------------------------------------------------------------------
    // @ Accessors
    // -----------------------------------------------------------------------------------------------------

    /**
     * Getter for data
     */
    get simulated_crypto_data$(): Observable<any> {
        return this._simulated_crypto_data.asObservable();
    }

    // -----------------------------------------------------------------------------------------------------
    // @ Public methods
    // -----------------------------------------------------------------------------------------------------

    /**
     * Get data
     */
    getSimulatedCryptoData(): Observable<any> {
        return this._httpClient.get('api/dashboards/crypto').pipe(
            tap((response: any) => {
                this._simulated_crypto_data.next(response);
            })
        );
    }

    getRealtimeScore(transactionId: number): Observable<HistoryScore> {
        const requestObj = {
            transaction_id: transactionId,
        };

        const options = { headers: this.headersRequest };

        return this._httpClient.post(`${this.baseUrl}/inductive/realtime-scoring`, requestObj, options)
        .pipe(
            map((result: {score: HistoryScore}) => {
                return result.score;
            })
        )
    }

    getRealLabelByTransaction(transactionId: number): Observable<string> {

        return this._httpClient.get(`${this.baseUrl}/elliptic/label/${transactionId}`)
        .pipe(
            map((result: string) => {
                return result;
            })
        )
        
    }
}
