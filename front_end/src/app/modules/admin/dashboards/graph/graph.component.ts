import { AsyncPipe, DecimalPipe, formatNumber, JsonPipe, PercentPipe } from '@angular/common';
import {
    ChangeDetectionStrategy,
    Component,
    OnDestroy,
    OnInit,
    ViewChild,
    ViewEncapsulation,
} from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatRippleModule } from '@angular/material/core';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';
import { MatTabsModule } from '@angular/material/tabs';
import { Router } from '@angular/router';
import { TranslocoModule } from '@jsverse/transloco';
import { ApexOptions, NgApexchartsModule } from 'ng-apexcharts';
import { BehaviorSubject, Subject } from 'rxjs';
import { GraphDashboardService } from './graph.service';
import { HistoryScore } from 'app/core/types/history_score.type';
import { KpiSummary, KpiSummaryModel } from 'app/core/types/kpi_scores.type';
import { RiskDistribution, RiskDistributionModel } from '../../../../core/types/riskDistribution.type';
import { MatPaginator, MatPaginatorModule } from '@angular/material/paginator';
import { SelectionModel } from '@angular/cdk/collections';
import { NetworkGraphComponent } from '../network-graph/network-graph.component';
import { ClusterAnalysisService } from './services/cluster_analysis.service';
import { ClusterAnalysis } from 'app/core/types/cluster_analysis.type';

@Component({
    selector: 'graph-analysis',
    templateUrl: './graph.component.html',
    styleUrls: ['./graph.component.scss'],
    encapsulation: ViewEncapsulation.None,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [
        TranslocoModule,
        MatIconModule,
        MatButtonModule,
        MatRippleModule,
        MatMenuModule,
        MatTabsModule,
        MatButtonToggleModule,
        NgApexchartsModule,
        MatTableModule,
        MatPaginatorModule,
        NetworkGraphComponent,
        AsyncPipe,
        PercentPipe
    ],
    providers: [DecimalPipe]
})
export class GraphComponent implements OnInit, OnDestroy {
    chartRiskDistribution: ApexOptions;
    chartClusterAnalysis: ApexOptions;
    riskDistribution: RiskDistribution;
    isKpiVisible: Boolean = false;
    isClusterVisible: Boolean = false;
    historyScoreDS = new MatTableDataSource<any>([]);
    selectedHistoryScore = new SelectionModel<any>(false, []);

    kpiSummary: BehaviorSubject<KpiSummary> = new BehaviorSubject(null);
    clusterAnalysis: BehaviorSubject<ClusterAnalysis> = new BehaviorSubject(null);

    private _unsubscribeAll: Subject<any> = new Subject<any>();

    @ViewChild(MatPaginator) set matPaginator(mp: MatPaginator) {
        this.historyScoreDS.paginator = mp;
    }

    /**
     * Constructor
     */
    constructor(
        private _router: Router,
        private graphDashboardService: GraphDashboardService,
        private clusterAnalysisService: ClusterAnalysisService,
        private decimalPipe: DecimalPipe
    ) { }

    ngOnDestroy(): void {
        // Unsubscribe from all subscriptions
        this._unsubscribeAll.next(null);
        this._unsubscribeAll.complete();
    }

    ngOnInit(): void {

        this.graphDashboardService.historyScores.subscribe(data => {

            if (!data) {
                return;
            }

            this.historyScoreDS.data = data;
            this.toggleRow(data[0]);
        });

        this.graphDashboardService.selectedScore.subscribe((selected: HistoryScore) => {

            if (!selected)
                return;

            this.clusterAnalysisService
                .getClusterAnalysisMetrics(selected?.transaction_id ?? 0)
                .subscribe((clusterAnalysis: ClusterAnalysis) => {

                    if (!clusterAnalysis)
                        return;

                    const labels = ['Max Risk', 'Mean Risk'];
                    const series = [
                        +this.decimalPipe.transform(clusterAnalysis.cluster_risk_max * 100, '1.2-2'),
                        +this.decimalPipe.transform(clusterAnalysis.cluster_risk_mean * 100, '1.2-2')
                    ];

                    this.loadClusterAnalysisChart(labels, series);
                    this.clusterAnalysis.next(clusterAnalysis);
                });
        });

        this.kpiSummary.subscribe((kpiData: KpiSummaryModel) => {

            if (!kpiData)
                return null;

            const series = [
                +formatNumber((kpiData?.highRiskPercent * 100), 'en-US', '1.2-2'),
                +formatNumber((kpiData?.mediumRiskPercent * 100), 'en-US', '1.2-2'),
                +formatNumber((kpiData?.lowRiskPercent * 100), 'en-US', '1.2-2')
            ];

            const labels = [
                'High Risk', 'Medium Risk', 'Low Risk'
            ];

            this.riskDistribution = new RiskDistributionModel(
                series,
                labels
            );

            this.loadRiskDistributionChart(labels, series);
        });


        this.graphDashboardService.getHistoricScoresMetrics(1800)
            .subscribe((scores: HistoryScore[]) => {
                this.isKpiVisible = true;

                const summary = KpiSummaryModel.fromTransactions(scores);

                this.kpiSummary.next(summary);
            });
    }

    // -----------------------------------------------------------------------------------------------------
    // @ Public methods
    // -----------------------------------------------------------------------------------------------------

    /**
     * Track by function for ngFor loops
     *
     * @param index
     * @param item
     */
    trackByFn(index: number, item: any): any {
        return item.id || index;
    }

    toggleRow(row: any) {
        this.selectedHistoryScore.toggle(row);
        this.graphDashboardService.setSelectedScore(row);
    }


    private loadRiskDistributionChart(labels: string[], series: number[]) {

        this.isClusterVisible = true;

        this.chartRiskDistribution = {
            chart: {
                animations: {
                    speed: 400,
                    animateGradually: {
                        enabled: false,
                    },
                },
                fontFamily: 'inherit',
                foreColor: 'inherit',
                height: '100%',
                type: 'donut',
                sparkline: {
                    enabled: true,
                },
            },
            colors: ['#EF4444', '#F6AD55', '#319795'],
            labels: labels,
            plotOptions: {
                pie: {
                    customScale: 0.9,
                    expandOnClick: false,
                    donut: {
                        size: '70%',
                    },
                },
            },
            series: series,
            states: {
                hover: {
                    filter: {
                        type: 'none',
                    },
                },
                active: {
                    filter: {
                        type: 'none',
                    },
                },
            },
            tooltip: {
                enabled: true,
                fillSeriesColor: false,
                theme: 'dark',
                custom: ({
                    seriesIndex,
                    w,
                }): string => `<div class="flex items-center h-8 min-h-8 max-h-8 px-3">
                                        <div class="w-3 h-3 rounded-full" style="background-color: ${w.config.colors[seriesIndex]};"></div>
                                        <div class="ml-2 text-md leading-none">${w.config.labels[seriesIndex]}:</div>
                                        <div class="ml-2 text-md font-bold leading-none">${w.config.series[seriesIndex]}%</div>
                                    </div>`
            },
        };

    }

    private loadClusterAnalysisChart(labels: string[], series: number[]) {

        // Task distribution
        this.chartClusterAnalysis = {
            chart: {
                fontFamily: 'inherit',
                foreColor: 'inherit',
                height: '100%',
                type: 'polarArea',
                toolbar: {
                    show: false,
                },
                zoom: {
                    enabled: false,
                },
            },
            labels: labels,
            legend: {
                position: 'bottom',
            },
            plotOptions: {
                polarArea: {
                    spokes: {
                        connectorColors: 'var(--fuse-border)',
                    },
                    rings: {
                        strokeColor: 'var(--fuse-border)',
                    },
                },
            },
            series: series,
            states: {
                hover: {
                    filter: {
                        type: 'darken',
                    },
                },
            },
            stroke: {
                width: 2,
            },
            theme: {
                monochrome: {
                    enabled: true,
                    color: '#93C5FD',
                    shadeIntensity: 0.75,
                    shadeTo: 'dark',
                },
            },
            tooltip: {
                followCursor: true,
                theme: 'dark',
            },
            yaxis: {
                labels: {
                    style: {
                        colors: 'var(--fuse-text-secondary)',
                    },
                },
            },
        };

        console.log('---- chartClusterAnalysis ----', this.chartClusterAnalysis);
    }
}