import { inject } from '@angular/core';
import { Routes } from '@angular/router';
import { ModelAnalysisComponent } from 'app/modules/admin/dashboards/analysis/analysis.component';
import { GraphDashboardService } from '../graph/graph.service';

export default [
    {
        path: '',
        component: ModelAnalysisComponent,
        resolve: {
            data: () => inject(GraphDashboardService).getHistoricScoresMetrics(1800),
        },
    },
] as Routes;
