/* eslint-disable */
import { FuseNavigationItem } from '@fuse/components/navigation';

export const defaultNavigation: FuseNavigationItem[] = [
    {
        id   : 'dashboard',
        title: 'Dashboard',
        type : 'basic',
        icon : 'heroicons_outline:chart-pie',
        link : '/dashboards/analysis'
    },
    {
        id   : 'scoring',
        title: 'Transfer',
        type : 'basic',
        icon : 'mat_solid:price_check',
        link : '/transfer'
    },
    {
        id   : 'attack',
        title: 'Attack Simulation',
        type : 'basic',
        icon : 'feather:shield-off',
        link : '/attack-simulation'
    }
];
