import { Component, AfterViewInit } from '@angular/core';
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import { NetworkGraphService } from './network-graph.service';

cytoscape.use(coseBilkent);

@Component({
  selector: 'app-network-graph',
  imports: [],
  templateUrl: './network-graph.component.html',
  styleUrl: './network-graph.component.scss'
})
export class NetworkGraphComponent implements AfterViewInit {

  cy: any;

  constructor(private networkGraphService: NetworkGraphService) {}

  ngAfterViewInit(): void {
    this.loadGraph();
  }

  
  loadGraph() {

    this.networkGraphService.getNetworkSubgraph(16754007).subscribe((data : {subgraph: []}) => {
      const elements = this.buildElements(data.subgraph);

      this.cy = cytoscape({
        container: document.getElementById('cy'),

        elements: elements,

        style: [
          {
            selector: 'node',
            style: {
              'background-color': 'data(color)',
              'label': 'data(label)',
              'width': 'mapData(risk, 0, 1, 20, 60)',
              'height': 'mapData(risk, 0, 1, 20, 60)',
              'font-size': 10,
              'text-valign': 'center',
              'color': '#000'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#9ca3af',
              'target-arrow-color': '#9ca3af',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier'
            }
          }
        ],

        layout: {
          name: 'cose-bilkent',
          // animate: true
        }
      });

      this.registerEvents();
    });
  }

  private buildElements(data: any) {

    const nodes = data.nodes.map(n => ({
      data: {
        id: n.transaction_id,
        label: `tx-${n.transaction_id}`,
        risk: n.risk,
        color: this.getRiskColor(n.risk)
      }
    }));

    const edges = data.edges.map(e => ({
      data: {
        id: e.source_transaction_id + '_' + e.target_transaction_id,
        source: e.source_transaction_id,
        target: e.target_transaction_id
      }
    }));

    return [...nodes, ...edges];
  }

  private getRiskColor(risk: number): string {

    if (risk < 0.30) return '#22c55e';   // green
    if (risk < 0.60) return '#facc15';   // yellow
    if (risk < 0.85) return '#fb923c';   // orange
    return '#ef4444';                    // red
  }

  private registerEvents() {

    this.cy.on('tap', 'node', (event) => {

      const node = event.target;

      const data = node.data();

      console.log('Transaction clicked:', data);

      // here you could open a details panel
    });
  }


}
