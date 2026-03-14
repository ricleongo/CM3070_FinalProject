import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
    providedIn: 'root'
})
export class NetworkGraphService {
    headersRequest = new HttpHeaders({ 'Content-Type': 'application/json' });
    private baseUrl = '/api/v1/transductive';

    constructor(private http: HttpClient) { }

    getNetworkSubgraph(transactionId: number) {
        const requestObj = {
            transaction_id: transactionId,
            hop_depth: 1
        };

        const options = { headers: this.headersRequest };

        return this.http.post(`${this.baseUrl}/network/subgraph`, requestObj, options);
    }
}
