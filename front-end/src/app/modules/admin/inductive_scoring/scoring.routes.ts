import { inject } from '@angular/core';
import { Routes } from '@angular/router';
import { ScoringComponent } from './scoring.component';
import { ScoringService } from './scoring.service';

export default [
    {
        path: '',
        component: ScoringComponent,
        resolve: {
            data: () => inject(ScoringService).getSimulatedCryptoData(),
        },
    },
] as Routes;
