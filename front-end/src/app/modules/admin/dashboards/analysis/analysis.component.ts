import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, ViewEncapsulation } from "@angular/core";
import { MatButtonModule } from "@angular/material/button";
import { MatRippleModule } from "@angular/material/core";
import { MatIconModule } from "@angular/material/icon";
import { MatMenuModule } from "@angular/material/menu";
import { MatTabsModule } from '@angular/material/tabs';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatTableModule } from '@angular/material/table';

import { ApexOptions, NgApexchartsModule } from 'ng-apexcharts';
import { Subject } from "rxjs";
import { Router } from "@angular/router";

@Component({
    selector: 'model-analysis',
    templateUrl: 'analysis.component.html',
    encapsulation: ViewEncapsulation.None,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [
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

    private _unsubscribeAll: Subject<any> = new Subject<any>();

    constructor(
        private _router: Router
    ) { }

    ngOnDestroy(): void {
        // Unsubscribe from all subscriptions
        this._unsubscribeAll.next(null);
        this._unsubscribeAll.complete();
    }

    ngOnInit(): void {
        const transductiveSeries = {
            name: 'Transductive (MD-GCN)',
            data: [87, 87, 87, 75, 81], // AUC, Precision, Recall, F1, Loss(inverted)
        };

        const inductiveSeries = {
            name: 'Transductive (MD-GCN)',
            data: [75, 16, 64, 25, 84], // AUC, Precision, Recall, F1, Loss(inverted)
        };

        const currentSeries = inductiveSeries

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

        // Attach SVG fill fixer to all ApexCharts
        window['Apex'] = {
            chart: {
                events: {
                    mounted: (chart: any, options?: any): void => {
                        this._fixSvgFill(chart.el);
                    },
                    updated: (chart: any, options?: any): void => {
                        this._fixSvgFill(chart.el);
                    },
                },
            },
        };
    }


    // -----------------------------------------------------------------------------------------------------
    // @ Private methods
    // -----------------------------------------------------------------------------------------------------

    /**
     * Fix the SVG fill references. This fix must be applied to all ApexCharts
     * charts in order to fix 'black color on gradient fills on certain browsers'
     * issue caused by the '<base>' tag.
     *
     * Fix based on https://gist.github.com/Kamshak/c84cdc175209d1a30f711abd6a81d472
     *
     * @param element
     * @private
     */
    private _fixSvgFill(element: Element): void {
        // Current URL
        const currentURL = this._router.url;

        // 1. Find all elements with 'fill' attribute within the element
        // 2. Filter out the ones that doesn't have cross reference so we only left with the ones that use the 'url(#id)' syntax
        // 3. Insert the 'currentURL' at the front of the 'fill' attribute value
        Array.from(element.querySelectorAll('*[fill]'))
            .filter((el) => el.getAttribute('fill').indexOf('url(') !== -1)
            .forEach((el) => {
                const attrVal = el.getAttribute('fill');
                el.setAttribute(
                    'fill',
                    `url(${currentURL}${attrVal.slice(attrVal.indexOf('#'))}`
                );
            });
    }

}