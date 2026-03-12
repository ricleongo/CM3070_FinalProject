import { Component, ViewEncapsulation } from '@angular/core';
import { NetworkGraphComponent } from '../network-graph/network-graph.component';


// import { DecimalPipe } from '@angular/common';
// import { MatButtonModule } from '@angular/material/button';
// import { MatButtonToggleModule } from '@angular/material/button-toggle';
// import { MatIconModule } from '@angular/material/icon';
// import { MatMenuModule } from '@angular/material/menu';
// import { MatTooltipModule } from '@angular/material/tooltip';


@Component({
    selector     : 'example',
    standalone   : true,
    templateUrl  : './example.component.html',
    encapsulation: ViewEncapsulation.None,
    imports: [
        NetworkGraphComponent
     ]
})
export class ExampleComponent
{
    /**
     * Constructor
     */
    constructor()
    {
    }
}
