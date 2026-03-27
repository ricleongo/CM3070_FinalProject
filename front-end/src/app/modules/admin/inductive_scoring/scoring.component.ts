import {
    AsyncPipe,
    CurrencyPipe,
    DecimalPipe,
    JsonPipe,
    NgClass,
    PercentPipe,
    UpperCasePipe,
} from '@angular/common';
import {
    ChangeDetectionStrategy,
    ChangeDetectorRef,
    Component,
    OnDestroy,
    OnInit,
    ViewChild,
    ViewEncapsulation,
} from '@angular/core';
import { FormsModule, ReactiveFormsModule, UntypedFormBuilder, UntypedFormGroup, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatOptionModule } from '@angular/material/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSidenavModule } from '@angular/material/sidenav';
import { FuseMediaWatcherService } from '@fuse/services/media-watcher';

import { FuseAlertComponent } from '@fuse/components/alert';

import { ApexOptions, ChartComponent, NgApexchartsModule } from 'ng-apexcharts';
import { BehaviorSubject, Subject, takeUntil } from 'rxjs';
import { ScoringService } from './scoring.service';
import { FuseDrawerComponent } from '@fuse/components/drawer';
import { MatMenuModule } from '@angular/material/menu';
import { BasicBeneficiaryInfo, BasicBeneficiaryInfoModel } from './types/basicInfo.type';
import { HistoryScore } from 'app/core/types/history_score.type';
import { MatProgressBarModule } from '@angular/material/progress-bar';

@Component({
    selector: 'scoring',
    templateUrl: './scoring.component.html',
    encapsulation: ViewEncapsulation.None,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [
        MatSidenavModule,
        MatIconModule,
        MatFormFieldModule,
        MatSelectModule,
        MatOptionModule,
        MatMenuModule,
        MatInputModule,
        MatButtonModule,
        MatProgressBarModule,

        NgApexchartsModule,
        FormsModule,
        ReactiveFormsModule,

        NgClass,

        UpperCasePipe,
        DecimalPipe,
        CurrencyPipe,
        PercentPipe,
        AsyncPipe,
        JsonPipe,

        FuseAlertComponent,
        FuseDrawerComponent
    ],
})
export class ScoringComponent implements OnInit, OnDestroy {
    @ViewChild('btcChartComponent') btcChartComponent: ChartComponent;
    @ViewChild('settingsDrawer') settingsDrawer: FuseDrawerComponent;

    watchlistChartOptions: ApexOptions = {};
    transferInfoForm: UntypedFormGroup;

    appConfig: any;
    btcOptions: ApexOptions = {};
    data: any;
    drawerMode: 'over' | 'side' = 'side';
    drawerOpened: boolean = true;
    btcToUsdRate: number = 8878.48;

    realtime_score: BehaviorSubject<HistoryScore> = new BehaviorSubject(null);
    real_label: BehaviorSubject<string> = new BehaviorSubject(null);

    private _unsubscribeAll: Subject<any> = new Subject<any>();

    /**
     * Constructor
     */
    constructor(
        private _scoringService: ScoringService,
        private _changeDetectorRef: ChangeDetectorRef,
        private _fuseMediaWatcherService: FuseMediaWatcherService,
        private _formBuilder: UntypedFormBuilder
    ) { }

    // -----------------------------------------------------------------------------------------------------
    // @ Lifecycle hooks
    // -----------------------------------------------------------------------------------------------------

    /**
     * On init
     */
    ngOnInit(): void {

        this.pupulateBasicForm(null);

        // Subscribe to media changes
        this._fuseMediaWatcherService.onMediaChange$
            .pipe(takeUntil(this._unsubscribeAll))
            .subscribe(({ matchingAliases }) => {
                // Set the drawerMode and drawerOpened if 'lg' breakpoint is active
                if (matchingAliases.includes('lg')) {
                    this.drawerMode = 'side';
                    this.drawerOpened = true;
                } else {
                    this.drawerMode = 'over';
                    this.drawerOpened = false;
                }

                // Mark for check
                this._changeDetectorRef.markForCheck();
            });

        // Get the data
        this._scoringService.simulated_crypto_data$
            .pipe(takeUntil(this._unsubscribeAll))
            .subscribe((data) => {
                // Store the data
                this.data = data;

                // Prepare the chart data
                this._prepareChartData();
            });

        this.realtime_score.subscribe((value) => {
            this.settingsDrawer?.toggle();
        });

    }

    /**
     * On destroy
     */
    ngOnDestroy(): void {
        // Unsubscribe from all subscriptions
        this._unsubscribeAll.next(null);
        this._unsubscribeAll.complete();
    }

    get bitcoinData() {
        return this.data?.watchlist[0] ?? {};
    }

    // -----------------------------------------------------------------------------------------------------
    // @ Public methods
    // -----------------------------------------------------------------------------------------------------

    pupulateBasicForm(basicInfo: BasicBeneficiaryInfo | null) {
        // Create the form
        this.transferInfoForm = this._formBuilder.group({
            account: [basicInfo?.account ?? '', [Validators.required, Validators.pattern(/^[0-9]{8,12}$/)]],
            fiatAmount: ['', [Validators.required, Validators.pattern(/^\d+(\.\d{1,2})?$/)]],
            btcAmount: ['', [Validators.required, Validators.pattern(/^\d+(\.\d{1,20})?$/)]],
            purpose: [''],
            beneficiaryName: [basicInfo?.beneficiaryName ?? '', [Validators.required, Validators.pattern(/^[a-zA-Z ]*$/)]],
            beneficiaryEmail: [basicInfo?.beneficiaryEmail ?? '', [Validators.required, Validators.email]],
        });

        this.transferInfoForm.get('fiatAmount')?.valueChanges.subscribe((usdValue) => {
            if (this.transferInfoForm.get('fiatAmount')?.dirty) {
                const btcCalculated = usdValue ? (Number(usdValue) / this.btcToUsdRate).toFixed(8) : '';
                this.transferInfoForm.patchValue({ btcAmount: btcCalculated }, { emitEvent: false });
            }
        });

        this.transferInfoForm.get('btcAmount')?.valueChanges.subscribe((btcValue) => {
            if (this.transferInfoForm.get('btcAmount')?.dirty) {
                const usdCalculated = btcValue ? (Number(btcValue) * this.btcToUsdRate).toFixed(2) : '';
                this.transferInfoForm.patchValue({ fiatAmount: usdCalculated }, { emitEvent: false });
            }
        });
    }

    scoreTransaction() {
        const transaction = this.transferInfoForm.value;

        this._scoringService.getRealtimeScore(transaction?.account)
            .subscribe((score: HistoryScore) => {
                this.realtime_score.next(score);
            });

        this._scoringService.getRealLabelByTransaction(transaction?.account)
            .subscribe((label: string) => {
                this.real_label.next(label);
            });
    }

    selectBeneficiary(beneficiaryId: number) {

        console.log(beneficiaryId);

        let beneficiary = null;

        switch (beneficiaryId) {
            case 0:
                beneficiary = new BasicBeneficiaryInfoModel(
                    80329479,
                    'Bernard Langley',
                    'bernardlangley@company.com'
                );
                break;
            case 1:
                beneficiary = new BasicBeneficiaryInfoModel(
                    158406298,
                    'Elsie Melendex',
                    'elsiemelendex@company.com'
                );
                break;
            case 2:
                beneficiary = new BasicBeneficiaryInfoModel(
                    188772009,
                    'Tina Harris',
                    'tinaharris@company.com'
                );
                break;

            case 3:
                beneficiary = new BasicBeneficiaryInfoModel(
                    147478192,
                    'Silva Foster',
                    'silvafoster@company.com'
                );
                break;
        };

        this.pupulateBasicForm(beneficiary);
    }

    // -----------------------------------------------------------------------------------------------------
    // @ Private methods
    // -----------------------------------------------------------------------------------------------------

    /**
     * Prepare the chart data from the data
     *
     * @private
     */
    private _prepareChartData(): void {
        // Watchlist options
        this.watchlistChartOptions = {
            chart: {
                animations: {
                    enabled: false,
                },
                width: '100%',
                height: '100%',
                type: 'line',
                sparkline: {
                    enabled: true,
                },
            },
            colors: ['#A0AEC0'],
            stroke: {
                width: 2,
                curve: 'smooth',
            },
            tooltip: {
                enabled: false,
            },
            xaxis: {
                type: 'category',
            },
        };

    }
}
