import { Component, ViewEncapsulation } from '@angular/core';
import { NetworkGraphComponent } from '../network-graph/network-graph.component';

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
