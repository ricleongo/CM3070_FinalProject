import { inject } from '@angular/core';
import { Routes } from '@angular/router';
import { ModelAnalysisComponent } from 'app/modules/admin/dashboards/analysis/analysis.component';

export default [
    {
        path: '',
        component: ModelAnalysisComponent,
        // resolve: {
        //     data: () => inject(ProjectService).getData(),
        // },
    },
] as Routes;
