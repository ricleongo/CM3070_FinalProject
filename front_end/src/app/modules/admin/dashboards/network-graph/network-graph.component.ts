import { Component, AfterViewInit, Input, OnInit } from '@angular/core';
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import { NetworkGraphService } from './network-graph.service';
import { GraphDashboardService } from '../graph/graph.service';
import { HistoryScore } from 'app/core/types/history_score.type';

cytoscape.use(coseBilkent);

@Component({
  selector: 'app-network-graph',
  imports: [],
  templateUrl: './network-graph.component.html',
  styleUrl: './network-graph.component.scss'
})
export class NetworkGraphComponent implements OnInit {

  cy: any;

  constructor(
    private networkGraphService: NetworkGraphService,
    private graphDashboardService: GraphDashboardService
  ) { }

  ngOnInit(): void {
    this.graphDashboardService.selectedScore.subscribe((selected: HistoryScore) => {
      this.loadGraph(selected?.transaction_id ?? 0);
    });
  }

  loadGraph(transactionID: number) {

    this.networkGraphService.getNetworkSubgraph(transactionID).subscribe((data: { subgraph: [] }) => {
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
              'font-size': 5,
              'text-valign': 'center',
              'color': 'data(textcolor)',
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
        label: n.transaction_id,
        risk: n.risk,
        color: this.getRiskColor(n.risk),
        textcolor: this.getRiskTextColor(n.risk)
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

  private getRiskTextColor(risk: number): string {

    if (risk < 0.30) return '#0e5d2b';   // green
    if (risk < 0.60) return '#6c580a';   // yellow
    if (risk < 0.85) return '#854e21';   // orange
    return '#8d2727';                    // red
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
