import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, ViewEncapsulation } from "@angular/core";
import { MatButtonModule } from "@angular/material/button";
import { MatRippleModule } from "@angular/material/core";
import { MatIconModule } from "@angular/material/icon";
import { MatMenuModule } from "@angular/material/menu";
import { MatTabsModule } from '@angular/material/tabs';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatTableModule } from '@angular/material/table';

import { ApexOptions, NgApexchartsModule } from 'ng-apexcharts';
import { BehaviorSubject, combineLatest, map, Observable, Subject } from "rxjs";
import { Router } from "@angular/router";

import { AnalysisService } from './analysis.service';
import { MLModelType } from "./model_type.enum";
import { EvaluationMetrics } from "./evaluation_metrics.type";
import { AsyncPipe, JsonPipe } from "@angular/common";
import { LossResults, LossResultsResponse } from "./loss_results.type";
import { BarSeries, LineSeries } from "./series_type";

@Component({
    selector: 'model-analysis',
    templateUrl: 'analysis.component.html',
    encapsulation: ViewEncapsulation.None,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [
        AsyncPipe,
        JsonPipe,
        MatIconModule,
        MatButtonModule,
        MatRippleModule,
        MatMenuModule,
        MatTabsModule,
        MatButtonToggleModule,
        NgApexchartsModule,
        MatTableModule,
    ]
})
export class ModelAnalysisComponent implements OnInit, OnDestroy {

    chartModelEvaluation: ApexOptions = {};
    chartTrainValLoss: ApexOptions = {};
    transductiveMetrics: BehaviorSubject<number[]> = new BehaviorSubject([]);
    inductiveMetrics: BehaviorSubject<number[]> = new BehaviorSubject([]);
    transductiveSpecialMetrics: BehaviorSubject<EvaluationMetrics> = new BehaviorSubject(null);
    inductiveSpecialMetrics: BehaviorSubject<EvaluationMetrics> = new BehaviorSubject(null);
    currentSpecialMetrics: BehaviorSubject<EvaluationMetrics> = new BehaviorSubject(null);
    modelType: BehaviorSubject<MLModelType> = new BehaviorSubject(MLModelType.Transductive);

    transductiveTrainLoss: BehaviorSubject<LossResults[]> = new BehaviorSubject(null);
    transductiveValLoss: BehaviorSubject<LossResults[]> = new BehaviorSubject(null);
    inductiveTrainLoss: BehaviorSubject<LossResults[]> = new BehaviorSubject(null);
    inductiveValLoss: BehaviorSubject<LossResults[]> = new BehaviorSubject(null);

    private _unsubscribeAll: Subject<any> = new Subject<any>();

    constructor(
        private _router: Router,
        private analysisService: AnalysisService
    ) { }

    ngOnDestroy(): void {
        // Unsubscribe from all subscriptions
        this._unsubscribeAll.next(null);
        this._unsubscribeAll.complete();
    }

    ngOnInit(): void {

        combineLatest([
            this.modelType,
            this.transductiveMetrics,
            this.inductiveMetrics
        ]).subscribe(([modelType, transductiveMetrics, inductiveMetrics]) => {

            const series = {
                name: (modelType == MLModelType.Transductive) ? 'Transductive (MD-GCN)' : 'Inductive (MD-GCN)',
                data: (modelType == MLModelType.Transductive) ? transductiveMetrics : inductiveMetrics
            }

            this._setup_radar_chart(series);
        });

        combineLatest([
            this.modelType,
            this.transductiveSpecialMetrics,
            this.inductiveSpecialMetrics
        ]).subscribe(([modelType, transductiveMetrics, inductiveMetrics]) => {
            if (modelType == MLModelType.Transductive) {
                this.currentSpecialMetrics.next(transductiveMetrics);
            } else {
                this.currentSpecialMetrics.next(inductiveMetrics);
            }
        });

        combineLatest([
            this.modelType,
            this.transductiveTrainLoss,
            this.transductiveValLoss,
            this.inductiveTrainLoss,
            this.inductiveValLoss
        ]).subscribe(([modelType, transductiveTrainLoss, transductiveValLoss, inductiveTrainLoss, inductiveValLoss]) => {

            if (modelType == MLModelType.Transductive) {
                const lossLabels = transductiveTrainLoss.map(item => item.epoch);

                const lossTrainSeries =
                    new LineSeries("Train Loss", transductiveTrainLoss.map(item => item.value));

                const lossValSeries =
                    new BarSeries("Validation Loss", transductiveValLoss.map(item => item.value));

                this._set_up_loss_chart(lossLabels, [lossTrainSeries, lossValSeries]);

            } else {
                const lossLabels = inductiveTrainLoss.map(item => item.epoch);

                const lossTrainSeries =
                    new LineSeries("Train Loss", inductiveTrainLoss.map(item => item.value));

                const lossValSeries =
                    new BarSeries("Validation Loss", inductiveValLoss.map(item => item.value));

                this._set_up_loss_chart(lossLabels, [lossTrainSeries, lossValSeries]);
            }

        });

        this._getGenericMetricsByType(MLModelType.Transductive).subscribe((metrics: number[]) => {
            this.transductiveMetrics.next(metrics);
        });

        this._getGenericMetricsByType(MLModelType.Inductive).subscribe((metrics: number[]) => {
            this.inductiveMetrics.next(metrics);
        });

        this._getSpecialMetricsByType(MLModelType.Transductive)
            .subscribe(metrics => {
                this.transductiveSpecialMetrics.next(metrics);
            });

        this._getSpecialMetricsByType(MLModelType.Inductive)
            .subscribe(metrics => {
                this.inductiveSpecialMetrics.next(metrics);
            });

        this._getLossResultsByType(MLModelType.Transductive)
            .subscribe(loss_results => {
                this.transductiveTrainLoss.next(loss_results.train_loss);
                this.transductiveValLoss.next(loss_results.val_loss);
            });

        this._getLossResultsByType(MLModelType.Inductive)
            .subscribe(loss_results => {
                this.inductiveTrainLoss.next(loss_results.train_loss);
                this.inductiveValLoss.next(loss_results.val_loss);
            });
    }


    // -----------------------------------------------------------------------------------------------------
    // @ Private methods
    // -----------------------------------------------------------------------------------------------------

    private _getGenericMetricsByType(modelType: MLModelType): Observable<number[]> {

        return this.analysisService.getEvaluationMetrics(modelType)
            .pipe(
                map((metrics: EvaluationMetrics) => {

                    const { loss, auc, precision, recall, f1 } = metrics;

                    const genericMetrics = Object.fromEntries(
                        // Order is important because match with the graph.
                        Object.entries({ auc, precision, recall, f1, loss })
                            .map(([key, value]) => [
                                key,
                                Number((value * 100).toFixed(2))
                            ])
                    );

                    return Object.values(genericMetrics) // get only values in order.
                })
            );

    }

    private _getSpecialMetricsByType(modelType: MLModelType): Observable<EvaluationMetrics> {

        return this.analysisService.getEvaluationMetrics(modelType)
            .pipe(
                map((metrics: EvaluationMetrics) => {

                    const { loss, auc, precision, recall, f1, fdr, nrc, far } = metrics;
                    const source = { loss, auc, precision, recall, f1, fdr, nrc, far };

                    // Map entries and convert to percentages with 2 decimals
                    const entries = Object.entries(source).map(([key, value]) => [
                        key,
                        Number((value * 100).toFixed(2))
                    ]);

                    // Reconstruct and cast to the interface
                    return Object.fromEntries(entries) as unknown as EvaluationMetrics;
                })
            );

    }

    private _getLossResultsByType(modelType: MLModelType): Observable<LossResultsResponse> {

        return this.analysisService.getLossResults(modelType)
            .pipe(
                map((loss_results: LossResultsResponse) => {

                    const formatLoss = (list: LossResults[]) =>
                        list.map(item => ({
                            ...item,
                            value: Number(item.value.toFixed(2))
                        }));


                    return {
                        train_loss: formatLoss(loss_results.train_loss),
                        val_loss: formatLoss(loss_results.val_loss)
                    };

                })
            );

    }

    private _setup_radar_chart(currentSeries: { name: string, data: number[] }) {

        this.chartModelEvaluation = {
            chart: {
                fontFamily: 'inherit',
                foreColor: 'inherit',
                type: 'radar',
                height: '100%',
                sparkline: {
                    enabled: true,
                }
            },
            dataLabels: {
                enabled: true,
                formatter: (val: number): string | number => `${val}%`,
                textAnchor: 'start',
                style: {
                    fontSize: '13px',
                    fontWeight: 500,
                },
                background: {
                    borderWidth: 0,
                    padding: 4,
                },
                offsetY: -15,
            },
            colors: ['#818CF8'],
            series: [
                currentSeries
            ],
            plotOptions: {
                radar: {
                    polygons: {
                        strokeColors: 'var(--fuse-border)',
                        connectorColors: 'var(--fuse-border)',
                    },
                },
            },
            markers: {
                strokeColors: '#818CF8',
                strokeWidth: 4,
                size: 5
            },
            fill: {
                opacity: 0.25
            },
            stroke: {
                width: 2
            },
            xaxis: {
                labels: {
                    show: true,
                    style: {
                        fontSize: '12px',
                        fontWeight: '500'
                    }
                },
                categories: ['AUC', 'Precision', 'Recall', 'F1 Score', 'Confidence \n(1-Loss)']
            },
            yaxis: {
                show: false,
                min: 0,
                max: 100
            },
            legend: {
                show: false,
                position: 'bottom'
            }
        };
    }

    private _set_up_loss_chart(chartLabels: string[], chartSeries: any) {

        this.chartTrainValLoss = {
            chart: {
                fontFamily: 'inherit',
                foreColor: 'inherit',
                height: '100%',
                type: 'line',
                toolbar: {
                    show: false,
                },
                zoom: {
                    enabled: false,
                },
            },
            colors: ['#166534', '#EEF2FF'],
            dataLabels: {
                enabled: true,
                enabledOnSeries: [0],
                background: {
                    borderWidth: 0,
                },
            },
            grid: {
                borderColor: 'var(--fuse-border)',
            },

            labels: chartLabels,
            series: chartSeries,

            legend: {
                show: false,
            },
            plotOptions: {
                bar: {
                    columnWidth: '50%',
                },
            },
            states: {
                hover: {
                    filter: {
                        type: 'darken',
                    },
                },
            },
            stroke: {
                width: [3, 0],
            },
            tooltip: {
                followCursor: true,
                theme: 'dark',
            },
            xaxis: {
                axisBorder: {
                    show: false,
                },
                axisTicks: {
                    color: 'var(--fuse-border)',
                },
                labels: {
                    style: {
                        colors: 'var(--fuse-text-secondary)',
                    },
                },
                tooltip: {
                    enabled: false,
                },
            },
            yaxis: {
                labels: {
                    offsetX: -16,
                    style: {
                        colors: 'var(--fuse-text-secondary)',
                    },
                },
            },
        };

    }

}