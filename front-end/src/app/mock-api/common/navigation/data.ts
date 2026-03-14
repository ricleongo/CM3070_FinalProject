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
        id   : 'transfer',
        title: 'Transfer',
        type : 'basic',
        icon : 'heroicons_outline:arrow-right-on-rectangle',
        link : '/transfer'
    },
    {
        id   : 'wallet',
        title: 'Crypto Wallet',
        type : 'basic',
        icon : 'heroicons_outline:wallet',
        link : '/wallet'
    },
    {
        id   : 'contributor',
        title: 'Contributor',
        type : 'basic',
        icon : 'heroicons_outline:adjustments-horizontal',
        link : '/example'
    }
];
